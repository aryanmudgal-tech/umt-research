"""Tests for web/pdf.py: the report printed to PDF by headless Chromium.

These drive a real browser, so they skip when Playwright's Chromium is not
installed (python -m playwright install chromium). Nothing goes to the
network: the print page and its assets are served from the repo by request
interception.
"""

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web import pdf

REPORT = (REPO_ROOT / "evals" / "fixtures" / "report_soil_pass.md").read_text()


def _chromium_available():
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium is not installed")


def test_the_print_page_typesets_every_equation():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = pdf.rendered_page(p, REPORT, "25 m beam on soil")
        try:
            assert page.locator(".katex-error").count() == 0
            assert page.locator(".katex-display").count() >= 10
            assert "$$" not in page.locator("#report").inner_text()
            assert page.title() == "25 m beam on soil"
        finally:
            browser.close()


def test_the_pdf_reads_as_the_report():
    from pypdf import PdfReader

    data = pdf.render_pdf(REPORT, "25 m beam on soil")
    assert data.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Key results" in text
    assert "Midspan deflection" in text and "3.015 mm downward" in text
    assert "$$" not in text and "\\," not in text


def test_assets_outside_static_are_refused():
    assert pdf.static_file("/static/styles.css") is not None
    assert pdf.static_file("/static/../app.py") is None
    assert pdf.static_file("/static/vendor/../../app.py") is None
