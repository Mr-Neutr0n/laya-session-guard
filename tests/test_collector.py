"""Tests for the local session collector and the candidate rules.

The collector reads real session logs, so these tests use synthetic log fixtures written
to a temporary directory. Nothing here reads the developer's real sessions, and nothing
here writes outside the temporary directory.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLLECTOR = ROOT / "review_kit" / "collect_from_sessions.py"
KIT = ROOT / "review_kit"
VALIDATOR = ROOT / "scripts" / "validate_review_set.py"


def load_collector():
    spec = importlib.util.spec_from_file_location("collector", COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pi_session(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def pi_session(user_text, tool_name, args):
    return [
        {"type": "message", "message": {"role": "user",
                                         "content": [{"type": "text", "text": user_text}]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": tool_name, "arguments": args}]}},
    ]


class RedactionGateTests(unittest.TestCase):
    """Drop, do not store. A candidate is only written if its text is clean."""

    def setUp(self):
        self.m = load_collector()

    def test_every_credential_family_is_covered(self):
        """One case per family, including the hyphenated forms that caused the audit finding."""
        cases = {
            "openai_key": "key sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "anthropic_key": "key sk-ant-0123456789abcdefghijklmnop",
            "stripe_key": "rk_live_0123456789abcdefghij",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key":
                "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "github_token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "gitlab_token": "glpat-ABCDEFGHIJKLMNOPQRST",
            "slack_token": "xoxb-123456789012-abcdefghijkl",
            "slack_webhook": "https://hooks.slack.com/services/T0000/B0000/XXXXXXXXXXXX",
            "bearer_token": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "basic_auth": "Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
            "private_key_block": "-----BEGIN RSA PRIVATE KEY----- here",
            "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
            "npm_token": "npm_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "google_api_key": "AIzaSyA0123456789abcdefghijklmnopqrstuv",
            "hex_blob": "blob " + "a1b2c3d4" * 8,
            "base64_blob": "blob A1b2C3d4E5f6G7h8I9j0KlMnOpQrStUvWxYz012345",
        }
        self.assertEqual(set(cases), set(self.m.CREDENTIAL_PATTERNS),
                         "a family is unlisted or a test is missing")
        import re as _re
        for family, text in cases.items():
            self.assertIsNotNone(_re.search(self.m.CREDENTIAL_PATTERNS[family], text),
                                 f"pattern {family} does not match its own sample")
            got = self.m.redact_reject_reason(text)
            self.assertIsNotNone(got, f"{family} passed the gate: {text[:40]}")
            self.assertTrue(got.startswith("credential:"), got)

    def test_hyphenated_and_prefixed_keys_all_drop(self):
        """The audit finding: [A-Za-z0-9]{20,} after "sk-" cannot match "sk-proj-"."""
        for text in ("sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH",
                     "sk-ant-AAAABBBBCCCCDDDDEEEEFFFF",
                     "sk-AAAABBBBCCCCDDDDEEEEFFFFGGGG",
                     "github_pat_11AAAABBBBCCCCDDDDEEEEFFFF",
                     "glpat-AAAABBBBCCCCDDDDEEEE"):
            self.assertIsNotNone(self.m.redact_reject_reason(text), text)

    def test_identity_strings_are_dropped(self):
        """identity_reject_reason reads the module global, so install the tokens on it."""
        original = self.m.IDENTITY_TOKENS
        self.m.IDENTITY_TOKENS = self.m.identity_tokens(["Samplegiven"])
        try:
            self.assertIn("samplegiven", self.m.IDENTITY_TOKENS)
            for text in ("committed by samplegiven today", "Signed, Samplegiven",
                         "samplegiven reviewed this"):
                self.assertIsNotNone(self.m.identity_reject_reason(text), text)
            self.assertIsNone(self.m.identity_reject_reason("an unrelated sentence"))
        finally:
            self.m.IDENTITY_TOKENS = original

    def test_common_words_are_not_identity_tokens(self):
        """An earlier derivation split the email and picked up 'gmail' and 'com'."""
        tokens = self.m.identity_tokens([])
        for common in ("com", "gmail", "http", "project", "update"):
            self.assertNotIn(common, tokens, f"'{common}' must not be an identity token")
        self.assertIsNone(self.m.redact_reject_reason("send it to gmail after the com merge"))

    def test_given_name_must_be_supplied_explicitly(self):
        """A given name is not derivable from a login, so --identity is the only route."""
        tokens = self.m.identity_tokens(["Quillpax"])
        self.assertIn("quillpax", tokens)
        self.assertIn("Quillpax", tokens)
        original = self.m.IDENTITY_TOKENS
        self.m.IDENTITY_TOKENS = tokens
        try:
            self.assertIsNotNone(self.m.identity_reject_reason("Thanks, Quillpax"))
        finally:
            self.m.IDENTITY_TOKENS = original

    def test_external_addresses_and_identifiers_are_dropped(self):
        self.assertEqual(self.m.redact_reject_reason("mail real.person@acmecorp.io"),
                         "email:external")
        self.assertEqual(self.m.redact_reject_reason("call +81 90 1234 5678"), "long_digits")
        self.assertEqual(self.m.redact_reject_reason("host 10.1.2.3 is up"), "ip_address")
        self.assertEqual(self.m.redact_reject_reason("blob " + "a1b2c3d4" * 8), "credential:hex_blob")

    def test_fictional_domains_are_allowed(self):
        for text in ("mail dev@example.com about it", "no address here at all"):
            self.assertIsNone(self.m.redact_reject_reason(text), text)

    def test_home_paths_are_replaced_not_dropped(self):
        """Matches the redaction checklist: paths reveal layout and are replaced."""
        cleaned = self.m.clean("edit /Users/someone/Developer/thing.py please")
        self.assertNotIn("/Users/someone", cleaned)
        self.assertIsNone(self.m.redact_reject_reason(cleaned))
        self.assertIn("/home/<user>", cleaned)

    def test_a_candidate_with_a_secret_is_dropped_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s" / "sess.jsonl"
            write_pi_session(path, pi_session(
                "please deploy this with AKIAIOSFODNN7EXAMPLE in it", "bash", {"command": "x"}))
            kept, stats, _, _ = self.m.collect(tmp, "", 10, 0)
        self.assertEqual(kept, [])
        credential_drops = {k: v for k, v in stats.items() if k.startswith("credential:")}
        self.assertTrue(credential_drops, f"the secret was not dropped at all: {dict(stats)}")

    def test_a_clean_candidate_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s" / "sess.jsonl"
            write_pi_session(path, pi_session(
                "please update the changelog for the release", "write",
                {"path": "src/changelog.md"}))
            kept, stats, _, _ = self.m.collect(tmp, "", 10, 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["kept"], 1)


class CandidateShapeTests(unittest.TestCase):
    def setUp(self):
        self.m = load_collector()
        self.kept, self.stats, _, _ = self.m.collect("", "", 0, 0)

    def _one(self, user_text, tool, args):
        case, reason = self.m.build_candidate("pi", "session-key", user_text, tool, args, 0)
        return case, reason

    def test_read_only_commands_are_not_decision_points(self):
        case, reason = self._one("please check the file for me now", "bash", {"command": "cat x.py"})
        self.assertIsNone(case)
        self.assertEqual(reason, "read_only_command")

    def test_non_decision_tools_are_rejected(self):
        case, reason = self._one("please read the file now", "read", {"path": "a.py"})
        self.assertIsNone(case)
        self.assertEqual(reason, "not_a_decision_tool")

    def test_a_candidate_with_no_preceding_user_turn_is_rejected(self):
        case, reason = self._one("", "write", {"path": "a.py"})
        self.assertIsNone(case)
        self.assertEqual(reason, "no_preceding_user_turn")

    def test_candidate_carries_no_label_and_no_annotations(self):
        case, _ = self._one("please update the changelog for the release", "write",
                            {"path": "src/changelog.md"})
        self.assertIsNotNone(case)
        self.assertIsNone(case["labels"])
        self.assertNotIn("annotations", case, "a candidate must not arrive pre-labelled")
        self.assertNotIn("adjudication", case)

    def test_candidate_is_tagged_pending_consent_everywhere(self):
        case, _ = self._one("please update the changelog for the release", "write",
                            {"path": "src/changelog.md"})
        self.assertEqual(case["provenance"]["origin"], "pending_consent")
        self.assertEqual(case["provenance"]["data_origin"], "pending_consent")
        self.assertEqual(case["provenance"]["approval_state"], "pending_consent")
        self.assertIsNone(case["provenance"]["consent_id"])
        self.assertEqual(case["provenance"]["redaction"], "applied")

    def test_candidate_notes_state_it_needs_approval(self):
        case, _ = self._one("please update the changelog for the release", "write",
                            {"path": "src/changelog.md"})
        notes = case["provenance"]["notes"].lower()
        self.assertIn("not consented", notes)
        self.assertIn("explicit approval", notes)
        self.assertIn("two independent human reviewers", notes)

    def test_candidate_keeps_only_minimal_context(self):
        case, _ = self._one("please update the changelog for the release", "write",
                            {"path": "src/changelog.md"})
        self.assertEqual(len(case["state"]["events"]), 2)
        self.assertEqual(case["state"]["events"][0]["source"], "direct_user")
        self.assertLessEqual(len(case["state"]["authorized_task"]), 700)

    def test_case_id_is_stable_and_grouped_by_session(self):
        a, _ = self._one("please update the changelog for the release", "write", {"path": "a.md"})
        b, _ = self.m.build_candidate("pi", "session-key", "please update the changelog for the release",
                                      "write", {"path": "a.md"}, 0)
        self.assertEqual(a["case_id"], b["case_id"])
        self.assertEqual(a["group"]["session_id"], b["group"]["session_id"])

    def test_case_id_matches_the_schema_pattern(self):
        import re
        case, _ = self._one("please update the changelog for the release", "write", {"path": "a.md"})
        self.assertRegex(case["case_id"], r"^[a-z0-9][a-z0-9-]{2,63}$")


class LocalOnlyTests(unittest.TestCase):
    def test_collector_makes_no_network_calls(self):
        """Bans network modules and calls, not URL text: a regex may legitimately name a
        Slack webhook host in order to recognise and drop one."""
        import ast
        tree = ast.parse(COLLECTOR.read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        for banned in ("socket", "urllib", "requests", "httpx", "http", "smtplib", "ftplib"):
            self.assertNotIn(banned, modules, f"collector must not import {banned}")
        source = COLLECTOR.read_text()
        for call in ("urlopen(", "urlretrieve(", "socket.create_connection("):
            self.assertNotIn(call, source, f"collector must not make a call: {call}")

    def test_collector_reads_only_the_session_roots(self):
        """Behavioural: a secret planted outside the roots must not reach the output."""
        self.assertIn(".pi", COLLECTOR.read_text())
        self.assertIn(".codex", COLLECTOR.read_text())
        for banned in (".ssh", ".aws", "Documents/identity"):
            self.assertNotIn(banned, COLLECTOR.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            write_pi_session(root / "s" / "sess.jsonl",
                             pi_session("please update the changelog for the release",
                                        "write", {"path": "a.md"}))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "secrets.txt").write_text("AKIAIOSFODNN7EXAMPLE")
            out = Path(tmp) / "out.jsonl"
            result = subprocess.run(
                [sys.executable, str(COLLECTOR), "--pi-root", str(root),
                 "--codex-root", str(Path(tmp) / "empty"), "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out.read_text())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s" / "sess.jsonl"
            write_pi_session(path, pi_session("please update the changelog now", "write",
                                              {"path": "a.md"}))
            out = Path(tmp) / "out.jsonl"
            result = subprocess.run(
                [sys.executable, str(COLLECTOR), "--pi-root", str(Path(tmp) / "s"),
                 "--codex-root", str(Path(tmp) / "empty"), "--dry-run", "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("nothing written", result.stderr)
            self.assertFalse(out.exists())


class CandidateValidationTests(unittest.TestCase):
    """A pending candidate is held to redaction and coherence, not to human labelling."""

    def setUp(self):
        self.draft = KIT / "draft_batch.jsonl"
        if not self.draft.exists():
            self.skipTest("no draft batch; run the collector first")

    def validate(self, path):
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(path), "--min-test-per-stratum", "0"],
            capture_output=True, text=True)

    def test_the_shipped_draft_validates_clean(self):
        result = self.validate(self.draft)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("0 errors", result.stdout)

    def test_a_prelabelled_candidate_is_rejected(self):
        rows = [json.loads(l) for l in self.draft.read_text().splitlines()[:1]]
        rows[0]["annotations"] = [
            {"reviewer_id": "r-a", "role": "human", "action": "allow", "rationale": "looks fine"},
            {"reviewer_id": "r-b", "role": "human", "action": "allow", "rationale": "fine"}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(rows[0]) + "\n")
            path = Path(handle.name)
        try:
            result = self.validate(path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("candidate.prelabelled", result.stdout)
        finally:
            path.unlink()

    def test_a_candidate_claiming_a_consent_id_is_rejected(self):
        rows = [json.loads(l) for l in self.draft.read_text().splitlines()[:1]]
        rows[0]["provenance"]["consent_id"] = "consent-invented"
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(rows[0]) + "\n")
            path = Path(handle.name)
        try:
            result = self.validate(path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("consent.missing", result.stdout)
        finally:
            path.unlink()

    def test_a_candidate_claiming_synthetic_provenance_is_rejected(self):
        rows = [json.loads(l) for l in self.draft.read_text().splitlines()[:1]]
        rows[0]["provenance"]["data_origin"] = "synthetic"
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(rows[0]) + "\n")
            path = Path(handle.name)
        try:
            result = self.validate(path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("consent.data_origin", result.stdout)
        finally:
            path.unlink()

    def test_a_candidate_without_approval_state_is_rejected(self):
        rows = [json.loads(l) for l in self.draft.read_text().splitlines()[:1]]
        rows[0]["provenance"].pop("approval_state")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(rows[0]) + "\n")
            path = Path(handle.name)
        try:
            result = self.validate(path)
            self.assertEqual(result.returncode, 1)
            self.assertIn("consent.approval_state", result.stdout)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
