"""End-to-end test of the professor's page in a real browser.

A real Chromium drives the real page against the real server, with the demo
pipeline standing in for Gemini (a recorded run replayed, the beam re-solved
and the gate re-run). Skips when Playwright's Chromium is not installed.
"""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evals.test_pdf import _chromium_available  # noqa: E402

pytestmark = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium is not installed")

SAMPLE = REPO_ROOT / "web" / "sample_brief.md"

# what each stage's animation is labelled, in the order the stages run
SCENE_LABELS = [
    "The brief being read and its facts picked out",
    "The beam being split into short pieces",
    "The beam deflecting under its load",
    "The checks being ticked off",
    "Two independent answers being compared",
    "The report being written",
]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from web.app import create_app
    from web.demo import demo_pipeline
    from web.storage import LocalStore

    store = LocalStore(tmp_path_factory.mktemp("runs"))
    app = create_app(
        store=store,
        pipeline=demo_pipeline(speed=0),
        render_pdf=lambda report_md, title: b"%PDF-1.4 test",
        load_key=False,
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not uv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def finished_page(server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(server)
        # record every scene the animation panel shows, in order
        page.evaluate(
            """() => {
              window.__scenes = [];
              new MutationObserver(() => {
                const svg = document.querySelector('#stage-panel svg');
                const label = svg && svg.getAttribute('aria-label');
                if (label && window.__scenes[window.__scenes.length - 1] !== label) window.__scenes.push(label);
              }).observe(document.getElementById('stage-panel'), {childList: true});
            }"""
        )
        page.set_input_files("#file", str(SAMPLE))
        page.wait_for_selector("#preview .katex, #preview h1")
        preview_text = page.inner_text("#preview")
        page.click("#run")
        page.wait_for_url("**/#/run/*", timeout=90_000)
        page.wait_for_selector("#report .katex-display", timeout=30_000)
        yield page, errors, preview_text
        browser.close()


def test_the_brief_preview_is_formatted_not_raw(finished_page):
    _page, _errors, preview = finished_page
    assert "Brief: 25 m beam on soil" in preview
    assert not preview.lstrip().startswith("#")


def test_every_stage_ran_and_showed_its_own_animation(finished_page):
    page, _errors, _ = finished_page
    assert page.locator("#stages .stage.done").count() == 6
    assert page.evaluate("window.__scenes") == SCENE_LABELS


def test_the_report_is_a_formatted_document(finished_page):
    page, _errors, _ = finished_page
    text = page.inner_text("#report")
    assert "Key results" in text and "3.015 mm downward" in text
    assert "$$" not in text
    assert not any(line.startswith(("#", "|")) for line in text.splitlines())
    assert page.locator("#report .katex-error").count() == 0
    assert page.locator("#report table").count() >= 3
    assert page.inner_text("#verdict").startswith("Verified")


def test_the_downloads_work(finished_page):
    page, _errors, _ = finished_page
    assert page.is_visible("#dl-pdf") and page.is_visible("#dl-md")
    origin = page.evaluate("location.origin")
    pdf = page.request.get(origin + page.get_attribute("#dl-pdf", "href"))
    assert pdf.ok and pdf.body().startswith(b"%PDF")
    md = page.request.get(origin + page.get_attribute("#dl-md", "href"))
    assert md.ok and "## Key results" in md.text()


def test_the_run_is_listed_and_reopens(finished_page):
    page, _errors, _ = finished_page
    item = page.locator("#runs a").first
    assert "25 m beam on soil" in item.inner_text() and "Verified" in item.inner_text()
    page.click("#new-brief")
    page.wait_for_selector("#drop", state="visible")  # a fresh upload, not the last brief
    assert not page.is_visible("#preview-wrap")
    item.click()
    page.wait_for_selector("#report .katex-display")
    assert "Key results" in page.inner_text("#report")


def test_how_the_answer_was_reached_reads_as_numbered_engineering_steps(finished_page):
    page, _errors, _ = finished_page
    page.click("#tab-how")
    page.wait_for_selector("#story > li >> nth=6")
    titles = page.locator("#story h3").all_inner_texts()
    assert titles == [
        "Read the brief", "Governing equation", "Finite element model", "Solution",
        "Verification checks", "Independent check", "Report",
    ]
    text = page.inner_text("#story")
    assert "Hermite cubic beam elements" in text
    assert "Supports 83.5 kN + foundation 666.5 kN = 750 kN = applied load." in text
    assert page.locator("#story .equation .katex").count() == 1  # the governing equation, typeset
    assert page.locator("#story .items li.skipped").count() == 2
    assert "Result confirmed" in text
    page.click("#tab-report")
    page.wait_for_selector("#report .katex-display", state="visible")


def test_a_past_run_can_be_renamed_and_deleted(finished_page):
    page, _errors, _ = finished_page
    page.click("#runs .run-more")
    page.click(".run-menu >> text=Rename")
    page.fill(".rename input", "Soil beam, first try")
    page.keyboard.press("Enter")
    page.wait_for_selector("#runs a >> text=Soil beam, first try")

    page.click("#runs .run-more")
    page.click(".run-menu >> text=Delete")
    assert "Delete \u201cSoil beam, first try\u201d?" in page.inner_text(".confirm")
    page.click(".confirm button.btn-danger")
    page.wait_for_selector("#runs-empty", state="visible")
    assert page.locator("#runs li").count() == 0


def test_no_script_errors(finished_page):
    _page, errors, _ = finished_page
    assert errors == []
