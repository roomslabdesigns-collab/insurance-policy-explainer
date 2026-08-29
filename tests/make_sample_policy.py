"""
Generates a small synthetic insurance policy PDF so the extraction pipeline
has something realistic to run against before you plug in a real policy.

NOT a real insurance policy — clause numbers and wording are made up purely
for testing (Phase 2 extraction, Phase 3 clause detection, Phase 4-5
retrieval, etc. all need *some* structured document to exercise against).

Run directly to (re)create the sample:
    python tests/make_sample_policy.py
"""

from pathlib import Path

import fitz  # PyMuPDF

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4, in points
MARGIN = 50

# One entry per page. `None` produces a page with NO text at all, to
# exercise the "empty" extraction_status path (simulating a scanned page).
PAGES = [
    "ABC GENERAL INSURANCE COMPANY\n\nHEALTH SHIELD POLICY\n\nPolicy Document\n\n"
    "Policy Year: 2025\nPolicy Number: HS-2025-0007",

    "SECTION 1: DEFINITIONS\n\n"
    "1.1 \"Insured Person\" means the person named in the policy schedule who is "
    "covered under this policy.\n\n"
    "1.2 \"Pre-Existing Disease\" means any condition, ailment, injury or disease "
    "diagnosed by a physician within 48 months prior to the effective date of "
    "the policy.\n\n"
    "1.3 \"Waiting Period\" means the time period during which specified "
    "illnesses/treatments are not covered.",

    "SECTION 2: SCOPE OF COVER\n\n"
    "2.1 Hospitalisation Expenses: The Company shall indemnify Medically "
    "Necessary Hospitalisation expenses incurred by the Insured Person during "
    "the Policy Period.\n\n"
    "2.2 Day Care Treatment: Medical treatment and/or surgical procedures "
    "listed in Annexure A, requiring less than 24 hours of hospitalisation, "
    "are covered.\n\n"
    "2.3 Dental Treatment: Dental treatment is covered only if necessitated by "
    "an accidental bodily injury requiring hospitalisation. Routine dental "
    "treatment, including fillings, extractions, and dentures, is not covered "
    "under this Section.",

    "SECTION 3: WAITING PERIODS\n\n"
    "3.1 Pre-Existing Conditions: A waiting period of 48 (forty-eight) months "
    "of continuous coverage applies to any Pre-Existing Disease, measured from "
    "the first policy inception date.\n\n"
    "3.2 Initial Waiting Period: A waiting period of 30 days applies to all "
    "illnesses from the policy start date, except claims arising from "
    "accidents.\n\n"
    "3.3 Specific Illnesses: A waiting period of 24 months applies to "
    "treatment of cataract, hernia, and joint replacement surgery.",

    "SECTION 4: EXCLUSIONS\n\n"
    "4.1 Cosmetic Surgery: Expenses towards cosmetic or plastic surgery are "
    "excluded unless required to treat an accidental injury.\n\n"
    "4.2 Dental and Vision:\n"
    "(a) Routine dental check-ups and cleaning are excluded.\n"
    "(b) Spectacles, contact lenses, and hearing aids are excluded.\n"
    "(c) Dental treatment other than treatment necessitated by accidental "
    "injury requiring hospitalisation is excluded.\n\n"
    "4.3 Claims in First Policy Year: Except for claims arising from an "
    "accident, no claim shall be payable for any illness first diagnosed "
    "within 90 days of policy commencement.",

    None,  # intentionally blank -> simulates a scanned/image-only page

    "See Annexure A.",  # intentionally very short -> simulates a low-text page
]


def build_sample_policy(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rect = fitz.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)

    doc = fitz.open()
    try:
        for page_text in PAGES:
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            if page_text:
                page.insert_textbox(rect, page_text, fontsize=11, fontname="helv")
        doc.save(output_path)
    finally:
        doc.close()

    return output_path


if __name__ == "__main__":
    written_path = build_sample_policy()
    print(f"Sample policy written to: {written_path}")
