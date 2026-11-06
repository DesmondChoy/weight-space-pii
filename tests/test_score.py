import unittest

import numpy as np

from wsp.data import Doc, Span
from wsp.predict import load, predict, word_tags
from wsp.score import SG_LABELS, Grade, Tally, bootstrap, decode, grade, summarize

TEXT = "Dear Tan Wei Ming, your NRIC is S1234567D."


def at(sub: str, label: str, text: str = TEXT) -> Span:
    i = text.index(sub)
    return Span(i, i + len(sub), label)


NRIC = at("S1234567D", "national_id")
SURNAME = at("Tan", "last_name")
GIVEN = at("Wei Ming", "first_name")


def score(gold, pred, ignore=()):
    gr = grade(Doc("t", TEXT, tuple(gold), tuple(ignore)), pred, SG_LABELS)
    return gr.total, dict(gr.failures)


class Cases(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(score([NRIC], [NRIC]), (Tally(1, 0, 0), {}))

    def test_wrong_type_outside_scored_labels(self):
        self.assertEqual(score([NRIC], [at("S1234567D", "biometric_identifier")]),
                         (Tally(0, 0, 1), {"wrong_type": 1}))

    def test_wrong_type_inside_scored_labels(self):
        self.assertEqual(score([SURNAME], [at("Tan", "national_id")]), (Tally(0, 1, 1), {"wrong_type": 1}))

    def test_boundary_off_by_one_word(self):
        self.assertEqual(score([NRIC], [at("is S1234567D", "national_id")]),
                         (Tally(0, 1, 1), {"wrong_boundary": 1}))

    def test_missed(self):
        self.assertEqual(score([NRIC], []), (Tally(0, 0, 1), {"missed": 1}))

    def test_false_positive(self):
        self.assertEqual(score([NRIC], [NRIC, at("Dear", "first_name")]), (Tally(1, 1, 0), {"false_positive": 1}))

    def test_unscored_prediction_is_not_a_false_positive(self):
        self.assertEqual(score([NRIC], [NRIC, at("Dear", "occupation")]), (Tally(1, 0, 0), {}))

    def test_scored_prediction_on_unscored_gold_is_a_false_positive(self):
        self.assertEqual(score([at("Dear", "occupation")], [at("Dear", "first_name")]),
                         (Tally(0, 1, 0), {"false_positive": 1}))


class Ignore(unittest.TestCase):
    CARD = at("Wei Ming", "IDCARDNUM")

    def test_scored_prediction_in_an_ignore_span_counts_neither_way(self):
        self.assertEqual(score([NRIC], [NRIC, at("Wei Ming", "national_id")], [self.CARD]), (Tally(1, 0, 0), {}))

    def test_same_prediction_without_the_ignore_span_is_a_false_positive(self):
        self.assertEqual(score([NRIC], [NRIC, at("Wei Ming", "national_id")]),
                         (Tally(1, 1, 0), {"false_positive": 1}))

    def test_ignore_span_never_hides_a_miss(self):
        self.assertEqual(score([NRIC], [], [at("S1234567D", "IDCARDNUM")]), (Tally(0, 0, 1), {"missed": 1}))


class Alias(unittest.TestCase):
    def test_tax_id_on_an_nric_counts_as_national_id(self):
        self.assertEqual(score([NRIC], [at("S1234567D", "tax_id")]), (Tally(1, 0, 0), {}))

    def test_tax_id_is_strict_when_tax_id_is_scored(self):
        labels = SG_LABELS + ("tax_id",)
        gr = grade(Doc("t", TEXT, (NRIC,)), [at("S1234567D", "tax_id")], labels)
        self.assertEqual((gr.total, dict(gr.failures)), (Tally(0, 1, 1), {"wrong_type": 1}))

    def test_tax_id_needs_the_exact_span(self):
        self.assertEqual(score([NRIC], [at("is S1234567D", "tax_id")]), (Tally(0, 0, 1), {"wrong_boundary": 1}))

    def test_tax_id_false_alarm_on_an_nric_shape_is_a_false_positive(self):
        text = "IDs S1234567D and T7654321A."
        doc = Doc("t", text, (Span(4, 13, "national_id"),))
        for label in ("national_id", "tax_id"):
            gr = grade(doc, [Span(4, 13, label), Span(18, 27, label)], SG_LABELS)
            self.assertEqual((gr.total, dict(gr.failures)), (Tally(1, 1, 0), {"false_positive": 1}), label)

    def test_tax_id_gets_no_credit_on_a_non_nric_national_id(self):
        text = "SSN 123-45-6789."
        doc = Doc("t", text, (Span(4, 15, "national_id"),))
        gr = grade(doc, [Span(4, 15, "tax_id")], SG_LABELS)
        self.assertEqual((gr.total, dict(gr.failures)), (Tally(0, 0, 1), {"wrong_type": 1}))

    def test_tax_id_never_stands_in_for_a_name(self):
        self.assertEqual(score([SURNAME], [at("Tan", "tax_id")]), (Tally(0, 0, 1), {"wrong_type": 1}))


class NameMerge(unittest.TestCase):
    def test_a_surname_tagged_as_a_given_name_is_right(self):
        self.assertEqual(score([SURNAME], [at("Tan", "first_name")]), (Tally(1, 0, 0), {}))

    def test_names_are_one_category(self):
        gr = grade(Doc("t", TEXT, (SURNAME, GIVEN)), [at("Tan Wei Ming", "last_name")], SG_LABELS)
        self.assertEqual((set(gr.by_label), gr.total), ({"name", "national_id"}, Tally(3, 0, 0)))

    def test_names_stay_separate_when_both_are_scored(self):
        labels = ("first_name", "last_name", "national_id")
        gr = grade(Doc("t", TEXT, (SURNAME,)), [at("Tan", "first_name")], labels)
        self.assertEqual((gr.total, dict(gr.failures)), (Tally(0, 1, 1), {"wrong_type": 1}))


class NameSplit(unittest.TestCase):
    def test_multi_word_gold_counts_once_per_word(self):
        self.assertEqual(score([GIVEN], []), (Tally(0, 0, 2), {"missed": 2}))

    def test_one_predicted_span_and_one_span_per_word_score_the_same(self):
        per_word = [at("Wei", "first_name"), at("Ming", "first_name")]
        self.assertEqual(score([GIVEN], [GIVEN]), (Tally(2, 0, 0), {}))
        self.assertEqual(score([GIVEN], per_word), (Tally(2, 0, 0), {}))

    def test_second_given_name_typed_as_surname(self):
        pred = [at("Wei", "first_name"), at("Ming", "last_name")]
        self.assertEqual(score([GIVEN], pred), (Tally(2, 0, 0), {}))

    def test_national_id_is_not_split(self):
        text = "ID S1234567D T7654321A"
        gold = [Span(3, 22, "national_id")]
        doc = Doc("t", text, (Span(3, 22, "national_id"),))
        self.assertEqual(grade(doc, [Span(3, 12, "national_id")], SG_LABELS).total, Tally(0, 1, 1))


class Decode(unittest.TestCase):
    WORDS = [(0, 3), (4, 7), (8, 12), (13, 15), (16, 25), (26, 30)]

    def test_bio_repair(self):
        tags = ["B-first_name", "I-first_name", "I-last_name", "O", "I-national_id", "I-national_id"]
        self.assertEqual(decode(self.WORDS, tags),
                         [Span(0, 7, "first_name"), Span(8, 12, "last_name"), Span(16, 30, "national_id")])

    def test_b_after_b_starts_a_new_entity(self):
        tags = ["B-last_name", "B-last_name", "O", "O", "O", "O"]
        self.assertEqual(decode(self.WORDS, tags), [Span(0, 3, "last_name"), Span(4, 7, "last_name")])

    def test_i_of_another_type_starts_a_new_entity(self):
        tags = ["B-first_name", "I-last_name", "O", "O", "O", "O"]
        self.assertEqual(decode(self.WORDS, tags), [Span(0, 3, "first_name"), Span(4, 7, "last_name")])


class Aggregate(unittest.TestCase):
    def test_micro_f1_and_bootstrap_agree(self):
        grades = [Grade({"national_id": Tally(1, 0, 0)}), Grade({"national_id": Tally(0, 1, 1)}),
                  Grade({"national_id": Tally(2, 0, 1)})]
        s = summarize(grades)
        self.assertEqual(s["overall"]["f1"], round(2 * 3 / (2 * 3 + 1 + 2), 4))
        b = bootstrap(grades, grades)
        self.assertAlmostEqual(b["f1"], 6 / 9)
        self.assertEqual((b["diff"], b["diff_se"]), (0.0, 0.0))
        self.assertGreater(b["se"], 0)

    def test_bootstrap_se_matches_jackknife(self):
        rng = np.random.default_rng(1)
        grades = [Grade({"x": Tally(*map(int, rng.integers(0, 4, size=3)))}) for _ in range(400)]
        c = np.array([[g.total.tp, g.total.fp, g.total.fn] for g in grades])
        loo = c.sum(axis=0) - c
        f = 2 * loo[:, 0] / (2 * loo[:, 0] + loo[:, 1] + loo[:, 2])
        jack = np.sqrt((len(f) - 1) / len(f) * ((f - f.mean()) ** 2).sum())
        self.assertAlmostEqual(bootstrap(grades)["se"] / jack, 1, delta=0.1)


class RealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = load()

    def test_word_takes_the_tag_of_its_first_subtoken(self):
        text = "Kow paid"
        enc = self.tok([text], return_offsets_mapping=True, return_tensors="pt")
        word_ids = enc.word_ids(0)
        self.assertEqual(word_ids[1:3], [0, 0], "Kow should split into two subtokens")
        l2i = self.model.config.label2id
        argmax = [l2i["O"]] * len(word_ids)
        argmax[1], argmax[2] = l2i["B-first_name"], l2i["B-last_name"]
        words, tags = word_tags(enc, 0, argmax, self.model.config.id2label)
        self.assertEqual(words[0], (0, 3))
        self.assertEqual(tags[0], "B-first_name")

    def test_batched_prediction_matches_one_text_at_a_time(self):
        texts = ["Claimant Rajesh s/o Kumaran (S7654321A) can be reached on +65 8234 5678.",
                 "Hi Mei.", "Dear Mr Tan Wei Ming, your policy for Siti binte Ahmad, NRIC T0234567J, is renewed.",
                 "Email zoë.ñúñez@example.com about the claim filed on 3 May 2024 by Jemina Sorgente."]
        batched = predict(self.tok, self.model, texts, batch=3)
        self.assertEqual(batched, [predict(self.tok, self.model, [t])[0] for t in texts])

    def test_predicted_entities_start_and_end_on_word_edges(self):
        tok, model = self.tok, self.model
        text = "Claimant Rajesh s/o Kumaran (S7654321A) can be reached on +65 8234 5678."
        spans = predict(tok, model, [text])[0]
        self.assertTrue(spans)
        for s in spans:
            v = text[s.start:s.end]
            self.assertEqual(v, v.strip())
            self.assertTrue(s.start == 0 or not text[s.start - 1].isalnum(), v)
            self.assertTrue(s.end == len(text) or not text[s.end].isalnum(), v)
        self.assertIn(Span(text.index("+65"), text.index("5678") + 4, "phone_number"), spans)


if __name__ == "__main__":
    unittest.main()
