"""Start the editor's kernel inside a macOS sandbox, send it code, and enforce the time and output limits."""

import json
import os
import secrets
import select
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wsp.harness.box import Box

PYTHON = Path(sys.executable)
SRT_PACKAGE = "@anthropic-ai/sandbox-runtime@0.0.81"
STARTUP_TIMEOUT = 120
MAX_REPLY_BYTES = 16 << 20


def srt_command() -> list[str]:
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx not found; the kernel sandbox runs through Anthropic's sandbox-runtime (srt)")
    return [npx, "-y", SRT_PACKAGE]


def settings(box: Box) -> dict:
    """srt's config. srt starts from a deny-all Seatbelt profile with no network, Apple Events, Launch Services or
    GPU; reads are open unless denied, so the home folder and temp areas are denied and the box re-allowed."""
    # .venv/bin/python resolves through uv's alias folders, so the whole uv python root must be readable.
    uv_python = PYTHON.resolve().parents[2]
    readable = [Path(sys.prefix), uv_python, box.model, box.practice, box.work, box.kernel, box.tmp]
    return {
        "network": {"allowedDomains": [], "deniedDomains": []},
        "filesystem": {
            "denyRead": [str(Path.home()), "/Users", "/Volumes", "/private/tmp", "/private/var/folders",
                         "/System/Volumes/Data"],
            "allowRead": [str(p) for p in readable],
            "allowWrite": [str(box.work), str(box.tmp)],
            "denyWrite": [],
        },
    }


def _is_reply(reply: dict, nonce: str) -> bool:
    return (reply.get("nonce") == nonce and isinstance(reply.get("output"), str)
            and type(reply.get("total")) is int and type(reply.get("seconds")) in (int, float))


@dataclass(frozen=True)
class KernelResult:
    output: str
    seconds: float
    restarted: bool


class Kernel:
    """One sandboxed kernel process. Variables persist between `run` calls until a restart."""

    def __init__(self, box: Box, log_path: Path, timeout: float = 900, max_output: int = 10_000):
        self.box, self.log_path, self.timeout, self.max_output = box, log_path, timeout, max_output
        self.proc: subprocess.Popen | None = None
        self.settings_path = box.root / "srt-settings.json"

    def start(self) -> None:
        self.settings_path.write_text(json.dumps(settings(self.box), indent=1))
        command = srt_command()
        req_r, self._req_w = os.pipe()
        self._rep_r, rep_w = os.pipe()
        # srt does not pass extra fds through, so the protocol uses the kernel's stdin and stdout.
        node_dir = str(Path(shutil.which("node") or command[0]).parent)
        env = {"HOME": str(self.box.tmp), "TMPDIR": str(self.box.tmp),
               "PATH": f"{node_dir}:/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8",
               "npm_config_cache": str(Path.home() / ".npm")}
        with self.log_path.open("ab") as log:
            self.proc = subprocess.Popen(
                [*command, "--settings", str(self.settings_path), "--", str(PYTHON), "-I",
                 str(self.box.kernel / "kernel.py"), str(self.box.root)],
                stdin=req_r, stdout=rep_w, stderr=log, cwd=self.box.work, env=env, start_new_session=True)
        os.close(req_r)
        os.close(rep_w)
        self._buf = b""
        reply = self._read(STARTUP_TIMEOUT)
        if not (isinstance(reply, dict) and reply.get("ready")):
            self._kill()
            raise RuntimeError(f"kernel failed to start; see {self.log_path}")

    def _read(self, timeout: float) -> dict | Literal["timeout", "died", "garbled"]:
        """The next reply line as a JSON object, or why there is none."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            if len(self._buf) > MAX_REPLY_BYTES:
                return "garbled"
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self._rep_r], [], [], left)[0]:
                return "timeout"
            chunk = os.read(self._rep_r, 1 << 20)
            if not chunk:
                return "died"
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        try:
            reply = json.loads(line)
        except (ValueError, RecursionError):
            return "garbled"
        return reply if isinstance(reply, dict) else "garbled"

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # macOS answers EPERM when the group holds only the dead, unreaped kernel.
            pass
        self.proc.wait()
        self.proc = None
        for fd in (self._req_w, self._rep_r):
            os.close(fd)

    def run(self, code: str) -> KernelResult:
        start = time.monotonic()
        nonce = secrets.token_hex(16)
        data = (json.dumps({"nonce": nonce, "code": code}) + "\n").encode()
        try:
            while data:
                data = data[os.write(self._req_w, data):]
        except BrokenPipeError:
            pass
        reply = self._read(self.timeout)
        if isinstance(reply, dict) and not _is_reply(reply, nonce):
            reply = "garbled"
        if not isinstance(reply, dict):
            seconds = time.monotonic() - start
            self._kill()
            self.start()
            what = {"timeout": f"The run was stopped after {seconds:.0f} seconds, the limit",
                    "died": "The kernel died during the run",
                    "garbled": "The kernel's reply was garbled, which happens when code writes to the kernel's "
                               "own reply pipe"}[reply]
            return KernelResult(f"{what}. The kernel was restarted, so all variables are lost.", seconds, True)
        return KernelResult(self._truncate(reply["output"], reply["total"]), reply["seconds"], False)

    def _truncate(self, output: str, total: int) -> str:
        if total <= self.max_output:
            return output
        return output[:self.max_output] + f"\n[output cut at {self.max_output:,} of {total:,} characters]"

    def close(self) -> None:
        self._kill()
