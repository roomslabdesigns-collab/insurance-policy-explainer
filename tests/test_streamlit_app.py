"""
Phase 9 verification script.

Run:
    python tests/test_streamlit_app.py

Uses Streamlit's official AppTest framework to drive the actual app.py
script (no browser needed) through the key user flows: initial state,
asking a question with an already-processed policy active, and the
invalid-PDF error path. Requires the llama.cpp server to be running for
the live-answer checks (skipped with a note otherwise).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from app.llm import is_server_available
from app.pdf_processing import InvalidPDFError
from app.rag import list_processed_policies
from app.ui.state import process_uploaded_policy

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def check(label: str, condition: bool) -> bool:
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    return condition


class FakeUploadedFile:
    """Minimal stand-in for Streamlit's UploadedFile -- lets us test
    process_uploaded_policy() directly without driving the real widget
    (AppTest does not support simulating file_uploader interactions)."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_initial_state() -> bool:
    print("=== Initial state (fresh session) ===")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    passed = check("App runs without an exception", not at.exception)

    # NOTE: this dev environment already has policies processed on disk
    # from earlier phases' testing, so the sidebar correctly auto-activates
    # the most recently processed one (a deliberate UX choice, not a bug --
    # a returning user shouldn't have to re-select every session). The
    # true "zero policies ever processed" empty state (`st.info("Upload
    # and process...")` in app.py) is a trivial, directly-inspectable
    # branch and isn't separately driven through AppTest here.
    if list_processed_policies():
        passed &= check(
            "A previously-processed policy auto-activated (Policy Information shown)",
            any("Indexed chunks" in m.value for m in at.markdown),
        )
    else:
        passed &= check(
            "Shows the 'upload a policy' prompt (no policies exist yet)",
            any("Upload and process" in i.value for i in at.info),
        )
    print()
    return passed


def test_invalid_pdf_upload() -> bool:
    print("=== Invalid PDF upload is rejected cleanly (Test 4) ===")
    passed = True
    try:
        process_uploaded_policy(
            FakeUploadedFile("not_a_real.pdf", b"this is not a pdf file"), "Test Policy", "2025"
        )
        passed &= check("InvalidPDFError was raised for garbage content", False)
    except InvalidPDFError as exc:
        passed &= check(f"InvalidPDFError raised cleanly: {exc}", True)
    print()
    return passed


def test_question_flow(document_id: str) -> bool:
    print("=== Test 1: Normal flow — ask a supported question ===")
    at = AppTest.from_file(APP_PATH)
    at.session_state["active_document_id"] = document_id
    at.run(timeout=60)
    passed = check("App runs without an exception", not at.exception)

    if not is_server_available():
        print("  LLM server not running -- skipping live question flow.\n")
        return passed

    example_buttons = [b for b in at.button if b.label == "Is dental treatment covered?"]
    passed &= check("Example question button is present", bool(example_buttons))
    if example_buttons:
        example_buttons[0].click().run(timeout=30)
        ask_buttons = [b for b in at.button if b.label == "Ask"]
        passed &= check("Ask button is present", bool(ask_buttons))
        if ask_buttons:
            ask_buttons[0].click().run(timeout=60)
            passed &= check("App still runs without an exception after asking", not at.exception)
            headings = [h.value for h in at.markdown if h.value.startswith("####")]
            passed &= check(f"An answer status card rendered: {headings}", bool(headings))

            print("=== Test 2: Unsupported question abstains safely ===")
            at2 = AppTest.from_file(APP_PATH)
            at2.session_state["active_document_id"] = document_id
            at2.run(timeout=60)
            at2.text_input(key="question_input").set_value("Does this policy cover space travel?").run()
            [b for b in at2.button if b.label == "Ask"][0].click().run(timeout=60)
            passed &= check("App runs without an exception", not at2.exception)
            body_text = " ".join(w.value for w in at2.markdown) + " ".join(w.value for w in at2.get("write"))
            passed &= check(
                "Answer does NOT claim 'Explicitly Excluded' for an unsupported topic",
                "Explicitly Excluded" not in body_text,
            )
    print()
    return passed


def main() -> None:
    all_passed = True
    all_passed &= test_initial_state()
    all_passed &= test_invalid_pdf_upload()

    policies = list_processed_policies()
    if policies:
        all_passed &= test_question_flow(policies[0]["document_id"])
    else:
        print("No processed policies found — run tests/test_pdf_extraction.py first.")
        all_passed = False

    print("=" * 78)
    print("Phase 9 environment check:", "PASS" if all_passed else "FAIL")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
