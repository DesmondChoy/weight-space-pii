"""The editor loop: send the task, run each reply's block, reply with the result, and log everything."""

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wsp.data import ROOT, read_jsonl
from wsp.harness.box import RUNS_ROOT, SANDBOX_ROOT, Box, RunDir, _rmtree, open_dir, setup_box
from wsp.harness.edit import write_edit
from wsp.harness.kernel_host import Kernel
from wsp.harness.prompt import render
from wsp.harness.server import RequestError, Result, Server

MAX_PYTHON_RUNS = 300
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
FENCE = re.compile(r"^```[ \t]*([^\n`]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
SYSTEM_PROMPT = "Answer the user's request."
CLI_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "DISABLE_AUTOUPDATER": "1"}
CLI_ENV_PASSED = ("HOME", "USER", "LOGNAME", "PATH", "LANG", "TMPDIR", "SHELL", "CLAUDE_CONFIG_DIR")
CONTEXT_CHECK = (
    "One last question, about your setup rather than the task. List every tool you can call, quote the "
    "instructions you were given (your system prompt, in full), and list any other context that reached you: "
    "hooks, plugins, skills, memory, CLAUDE.md files, an output style, or anything else. Report only what you "
    "can see, and say so when there is none. Do not guess.")


@dataclass(frozen=True)
class Python:
    code: str


@dataclass(frozen=True)
class Forward:
    raw: str


@dataclass(frozen=True)
class Final:
    text: str


@dataclass(frozen=True)
class Invalid:
    blocks: int


Reply = Python | Forward | Final | Invalid


def parse_reply(text: str) -> Reply:
    """Exactly one ```python or ```forward block is that block; none is a final answer; more is invalid."""
    blocks = []
    for info, body in FENCE.findall(text):
        lang = info.split()[0].lower() if info.split() else ""
        if lang in ("python", "forward"):
            blocks.append((lang, body.rstrip("\n")))
    if not blocks:
        return Final(text)
    if len(blocks) > 1:
        return Invalid(len(blocks))
    lang, body = blocks[0]
    return Python(body) if lang == "python" else Forward(body)


@dataclass(frozen=True)
class EditorReply:
    """One CLI call. `usage` is this call's; the CLI's total_cost_usd is the session's running total, so
    `cost` is this call's share of it and `total_cost` the total itself."""
    text: str
    usage: dict
    cost: float
    total_cost: float
    session_id: str
    seconds: float


class EditorLike(Protocol):
    session_id: str
    retries: list[dict]

    def send(self, message: str) -> EditorReply: ...

    def info(self) -> dict: ...


class Editor:
    """The editor model behind the Claude Code CLI with every tool off, one session resumed per message."""

    def __init__(self, cwd: Path, log: Callable, model: str = "claude-opus-5-5", effort: str = "max",
                 sleep: Callable[[float], None] = time.sleep, first_wait: float = 60, max_wait: float = 900,
                 total_wait: float = 6 * 3600):
        self.cwd, self.log, self.model, self.effort, self.sleep = cwd, log, model, effort, sleep
        self.first_wait, self.max_wait, self.total_wait = first_wait, max_wait, total_wait
        self.session_id = str(uuid.uuid4())
        self.started = False
        self.total_cost = 0.0
        self.retries: list[dict] = []

    def command(self, resume: bool) -> list[str]:
        return ["claude", "-p", "--model", self.model, "--effort", self.effort, "--tools", "",
                "--strict-mcp-config", "--disable-slash-commands", "--setting-sources", "",
                "--system-prompt", SYSTEM_PROMPT, "--output-format", "json",
                *(["--resume", self.session_id] if resume else ["--session-id", self.session_id])]

    def info(self) -> dict:
        return {"model": self.model, "effort": self.effort, "session_id": self.session_id, "cli_version": cli_version(),
                "cli_command": self.command(resume=False), "cli_env": CLI_ENV, "cli_env_passed": list(CLI_ENV_PASSED),
                "cli_cwd": str(self.cwd)}

    def _call(self, message: str) -> tuple[EditorReply | None, dict]:
        """The reply, or None and what went wrong, with any usage and cost the failed call reported."""
        resume = self.started or session_file(self.session_id) is not None
        start = time.monotonic()
        proc = subprocess.run(self.command(resume), input=message, capture_output=True, text=True,
                              cwd=self.cwd, env=cli_env())
        seconds = time.monotonic() - start
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out = None
        if not isinstance(out, dict):
            return None, {"error": f"exit {proc.returncode}; stdout {proc.stdout[-500:]!r}; "
                                   f"stderr {proc.stderr[-500:]!r}"}
        total = float(out.get("total_cost_usd") or self.total_cost)
        if proc.returncode != 0 or out.get("is_error") or out.get("subtype") != "success":
            return None, {"error": f"exit {proc.returncode}; {str(out.get('result'))[-500:]!r}",
                          "usage": out.get("usage"), "total_cost_usd": out.get("total_cost_usd")}
        return EditorReply(out.get("result") or "", out.get("usage") or {}, total - self.total_cost, total,
                           out.get("session_id", self.session_id), seconds), {}

    def send(self, message: str) -> EditorReply:
        """Send one message. A failed call is waited out and resent on the same session; it is not a turn."""
        wait, waited = self.first_wait, 0.0
        while True:
            reply, failure = self._call(message)
            if reply is not None:
                self.started = True
                self.total_cost = reply.total_cost
                return reply
            if waited + wait > self.total_wait:
                raise RuntimeError(f"the editor CLI kept failing for {waited:.0f} s: {failure['error']}")
            self.log("cli_retry", wait=wait, **failure)
            self.retries.append({"wait": wait, **failure})
            self.sleep(wait)
            waited += wait
            wait = min(wait * 2, self.max_wait)


def cli_env() -> dict[str, str]:
    """The CLI's whole environment. A launcher such as the Claude desktop app sets dozens of Claude variables,
    effort and compaction among them, and none of them may reach the run, so only CLI_ENV_PASSED is copied."""
    return {k: os.environ[k] for k in CLI_ENV_PASSED if k in os.environ} | CLI_ENV


def cli_version() -> str:
    return subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True,
                          env=cli_env()).stdout.strip()


def session_file(session_id: str) -> Path | None:
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    return next((config / "projects").glob(f"*/{session_id}.jsonl"), None)


def compacted(session_id: str) -> bool:
    path = session_file(session_id)
    return path is not None and "compact_boundary" in path.read_text()


class Transcript:
    def __init__(self, path: Path):
        self.f = path.open("a")

    def log(self, kind: str, **fields) -> None:
        self.f.write(json.dumps({"kind": kind, "time": time.time(), **fields}, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


@dataclass
class Loop:
    """The state of one editor session and the reply to each of the editor's messages."""
    box: Box
    rundir: RunDir
    kernel: Kernel
    server: Server
    transcript: Transcript
    budget: int
    passes_used: int = 0
    python_runs: int = 0
    kernel_restarts: int = 0
    failed_forwards: int = 0
    invalid_replies: int = 0
    turn: int = 0
    last_edit: tuple[list, str] | None = None

    @property
    def left(self) -> int:
        return self.budget - self.passes_used

    @property
    def final_choice(self) -> str:
        return (f"Make your final choice (the tensors in {self.box.edited} when you finish are submitted) and reply "
                "without a code block, saying what you changed, the evidence for it and why you expect it to help.")

    def respond(self, reply: Reply) -> str:
        match reply:
            case Python(code):
                return self._python(code)
            case Forward(raw):
                return self._forward(raw)
            case Final():
                return (f"Your reply had no ```python or ```forward block, so nothing was run. {self.left} forward "
                        "passes are left; the session continues until all are used.")
            case Invalid(blocks):
                self.invalid_replies += 1
                self.transcript.log("invalid", turn=self.turn, blocks=blocks)
                return f"Your reply had {blocks} code blocks. Reply with exactly one ```python or ```forward block. Nothing was run."

    def _python(self, code: str) -> str:
        if self.python_runs >= MAX_PYTHON_RUNS:
            return f"All {MAX_PYTHON_RUNS} Python runs are used, so nothing was run."
        self.python_runs += 1
        r = self.kernel.run(code)
        self.kernel_restarts += r.restarted
        self.transcript.log("python", turn=self.turn, n=self.python_runs, code=code, output=r.output,
                            seconds=r.seconds, restarted=r.restarted)
        out = r.output.rstrip("\n")
        return f"OUTPUT of Python run {self.python_runs} of at most {MAX_PYTHON_RUNS}:\n```\n{out}\n```"

    def _forward(self, raw: str) -> str:
        if self.left == 0:
            return f"No forward passes are left, so nothing was run. {self.final_choice}"
        k = self.passes_used + 1
        try:
            req = self.server.parse(raw)
            edit = self._snapshot_edit(req.model)
            result = self.server.handle(k, req)
        except RequestError as e:
            self.failed_forwards += 1
            self.transcript.log("forward_failed", turn=self.turn, request=raw, reason=str(e))
            return f"Forward request failed and was not counted: {e}\nForward passes left: {self.left}."
        except Exception as e:
            self.failed_forwards += 1
            self.transcript.log("forward_error", turn=self.turn, request=raw, traceback=traceback.format_exc())
            return (f"Forward request failed with an internal error ({type(e).__name__}) and was not counted.\n"
                    f"Forward passes left: {self.left}.")
        summary = result.summary
        self.passes_used = k
        self._save_forward(k, raw, result, edit)
        self.transcript.log("forward", turn=self.turn, k=k, request=raw, summary=summary)
        msg = (f"Forward pass {k} of {self.budget} done; per-item results in {summary['pred_file']}. Summary:\n"
               f"```\n{json.dumps(summary, indent=1)}\n```\nForward passes left: {self.left}.")
        if self.left == 0:
            msg += f"\n\nAll {self.budget} forward passes are used. {self.final_choice}"
        return msg

    def _snapshot_edit(self, model: str) -> dict:
        """The edit this pass is about to run, written to a scratch folder: the manifest, and edit.npz unless the
        edit is the one the last recorded pass ran. An "original" pass runs no edit."""
        scratch = self.rundir.root / ".edit_snapshot"
        _rmtree(scratch)
        if model == "original":
            return {"tensors": [], "n_tensors": 0, "entries_changed": 0, "npz": None}
        try:
            manifest = write_edit(self.box, self.box.weights, scratch)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "npz": None}
        if self.last_edit is not None and manifest["tensors"] == self.last_edit[0]:
            (scratch / "edit.npz").unlink()
            return {**manifest, "npz": self.last_edit[1]}
        return {**manifest, "npz": "new"}

    def _save_forward(self, k: int, raw: str, result: Result, edit: dict) -> None:
        d = self.rundir.forward / str(k)
        d.mkdir(parents=True)
        (d / "request.json").write_text(raw + "\n")
        (d / "summary.json").write_text(json.dumps(result.summary, indent=1) + "\n")
        (d / "pred.jsonl").write_text(result.pred, encoding="utf-8")
        if result.meta is not None:
            (d / "meta.json").write_text(json.dumps(result.meta, ensure_ascii=False) + "\n", encoding="utf-8")
        if edit["npz"] == "new":
            os.replace(self.rundir.root / ".edit_snapshot" / "edit.npz", d / "edit.npz")
            edit = {**edit, "npz": f"{k}/edit.npz"}
            self.last_edit = (edit["tensors"], edit["npz"])
        (d / "edit_manifest.json").write_text(json.dumps(edit, indent=1) + "\n")


WORK_NOTE_MAX_BYTES = 1 << 20
WORK_NOTES_MAX_BYTES = 20 << 20


def _work_note_skipped(rel: str, is_dir: bool) -> bool:
    """Files the run folder already holds elsewhere: the edit, each pass's predictions, and its arrays."""
    top = rel.split("/")[0]
    return top == "edited" or (is_dir and top.startswith("fwd_")) or (rel == top and top.startswith("pred_"))


def copy_work_notes(box: Box, dest: Path) -> dict:
    """Copy the small regular files in work/ to dest, never following a symlink or reading a hard link, since
    the kernel controls everything under work/. Returns what was copied and what was left out, and why."""
    copied, skipped, total = [], [], 0

    def walk(dir_fd: int, prefix: str, depth: int) -> None:
        nonlocal total
        for entry in sorted(os.scandir(dir_fd), key=lambda e: e.name):
            rel = prefix + entry.name
            if entry.is_symlink():
                skipped.append([rel, "symlink"])
            elif entry.is_dir(follow_symlinks=False):
                if _work_note_skipped(rel, True):
                    continue
                if depth >= 8:
                    skipped.append([rel, "too deep"])
                    continue
                fd = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
                try:
                    walk(fd, rel + "/", depth + 1)
                finally:
                    os.close(fd)
            elif not _work_note_skipped(rel, False):
                try:
                    fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
                except OSError:
                    skipped.append([rel, "cannot be opened"])
                    continue
                with os.fdopen(fd, "rb") as f:
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        skipped.append([rel, "not a plain file"])
                    elif st.st_size > WORK_NOTE_MAX_BYTES or total + st.st_size > WORK_NOTES_MAX_BYTES:
                        skipped.append([rel, f"{st.st_size} bytes"])
                    else:
                        data = f.read(st.st_size)
                        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                        (dest / rel).write_bytes(data)
                        copied.append(rel)
                        total += len(data)

    fd = open_dir(box.root, "work")
    try:
        walk(fd, "", 0)
    finally:
        os.close(fd)
    return {"copied": len(copied), "bytes": total, "skipped": skipped}


def _lock_sha256() -> str | None:
    path = ROOT / "lock.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git_head() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _write_edit(box: Box, rundir: RunDir) -> tuple[dict | None, str | None]:
    try:
        return write_edit(box, box.weights, rundir.edit), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run(name: str, budget: int, max_turns: int | None = None, effort: str = "max", ask_context: bool = False,
        make_editor: Callable[[Box, Callable], EditorLike] | None = None, box_root: Path = SANDBOX_ROOT,
        runs_root: Path = RUNS_ROOT, kernel_timeout: float = 900, overwrite: bool = False) -> dict:
    """Run one editor session to its final answer and write the run folder. Returns run.json's content.

    `max_turns` caps the editor's replies, counting the final one; None sets no cap. final_answer.md and
    run.json are written even when the session stops with an error, which is then re-raised."""
    assert max_turns is None or max_turns >= 2, "the turn limit must leave room for the wrap-up reply"
    start = time.monotonic()
    rundir = RunDir(runs_root / name)
    if rundir.root.exists():
        if not overwrite:
            raise FileExistsError(f"{rundir.root} exists; pass overwrite=True to replace it")
        shutil.rmtree(rundir.root)
    rundir.root.mkdir(parents=True)
    box = setup_box(name, fresh=True, root=box_root)
    transcript = Transcript(rundir.transcript)
    editor = make_editor(box, transcript.log) if make_editor else Editor(box.cli, transcript.log, effort=effort)
    kernel = Kernel(box, rundir.kernel_log, timeout=kernel_timeout)
    loop = Loop(box, rundir, kernel, Server(box, read_jsonl(box.practice_file)), transcript, budget)
    usage = dict.fromkeys(USAGE_KEYS, 0)
    cost = 0.0
    final, ended_by, edit, context, error = None, "error", None, None, None

    def send(message: str, kind: str = "assistant") -> str:
        nonlocal cost
        transcript.log("user", turn=loop.turn + 1, text=message)
        reply = editor.send(message)
        for key in USAGE_KEYS:
            usage[key] += int(reply.usage.get(key, 0))
        cost = reply.total_cost
        transcript.log(kind, turn=loop.turn + 1, text=reply.text, usage=reply.usage, cost=reply.cost,
                       total_cost=reply.total_cost, seconds=reply.seconds, session_id=reply.session_id)
        return reply.text

    try:
        kernel.start()
        message, wrapping = render(budget, box), False
        while True:
            text = send(message)
            loop.turn += 1
            if wrapping:
                final, ended_by = text, "turn_limit"
                break
            reply = parse_reply(text)
            if isinstance(reply, Final) and loop.left == 0:
                final, ended_by = text, "final_answer"
                break
            message = loop.respond(reply)
            if max_turns is not None and loop.turn == max_turns - 1:
                wrap = "The turn limit for this session is reached."
                if loop.final_choice not in message:
                    wrap += f" {loop.final_choice}"
                message = wrap if isinstance(reply, Final | Invalid) else f"{message}\n\n{wrap}"
                wrapping = True
        edit = _write_edit(box, rundir)
        if ask_context:
            context = send(CONTEXT_CHECK, kind="context_check")
            rundir.context_check.write_text(f"## Question\n\n{CONTEXT_CHECK}\n\n## Reply\n\n{context.rstrip()}\n")
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        transcript.log("error", traceback=traceback.format_exc())
        raise
    finally:
        kernel.close()
        _rmtree(rundir.root / ".edit_snapshot")
        if edit is None:
            edit = _write_edit(box, rundir)
        manifest, edit_error = edit
        rundir.final_answer.write_text(
            final.rstrip("\n") + "\n" if final is not None else f"[No final answer: the run stopped with {error}]\n")
        try:
            work_notes = copy_work_notes(box, rundir.root / "work_notes")
        except Exception as e:
            work_notes = {"error": f"{type(e).__name__}: {e}"}
        info = editor.info()
        session = session_file(info["session_id"])
        try:
            if session is not None:
                shutil.copyfile(session, rundir.root / "cli_session.jsonl")
        except OSError:
            session = None
        result = {
            **info, "budget": budget, "max_turns": max_turns, "turns": loop.turn, "ended_by": ended_by,
            "error": error, "python_runs": loop.python_runs, "passes_used": loop.passes_used,
            "failed_forwards": loop.failed_forwards, "invalid_replies": loop.invalid_replies,
            "kernel_restarts": loop.kernel_restarts, "usage": usage, "cost_usd": round(cost, 6),
            "wall_seconds": round(time.monotonic() - start, 1), "compacted": compacted(info["session_id"]),
            "cli_retries": editor.retries, "context_check": context is not None,
            "edit": manifest and {k: manifest[k] for k in ("n_tensors", "entries_changed")},
            "edit_error": edit_error, "work_notes": work_notes, "cli_session_copied": session is not None,
            "lock_sha256": _lock_sha256(), "git": _git_head(), "box": str(box.root),
        }
        rundir.run_json.write_text(json.dumps(result, indent=1) + "\n")
        transcript.close()
    return result
