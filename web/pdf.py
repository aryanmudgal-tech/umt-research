"""Print a report to PDF with headless Chromium, so the PDF matches the page.

The report is rendered by the same files the page uses (web/static/render.js,
styles.css, the vendored markdown-it and KaTeX), loaded into Chromium at a
made-up origin whose every request is answered from the repo: nothing goes to
the network, and nothing outside web/static can be read.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = (WEB_DIR / "static").resolve()
PRINT_TEMPLATE = WEB_DIR / "print.html"
ORIGIN = "http://report.local"
RENDER_TIMEOUT_MS = 30_000

FOOTER = (
    '<div style="width:100%;font:8px sans-serif;color:#7b8984;padding:0 16mm;display:flex;'
    'justify-content:space-between"><span>{title}</span>'
    '<span><span class="pageNumber"></span> of <span class="totalPages"></span></span></div>'
)


def static_file(url_path: str):
    """The file under web/static a /static/... path names, or None."""
    if not url_path.startswith("/static/"):
        return None
    candidate = (STATIC_DIR / url_path[len("/static/"):]).resolve()
    if STATIC_DIR not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _page_html(report_md: str, title: str) -> str:
    payload = json.dumps({"md": report_md, "title": title}).replace("</", "<\\/")
    return PRINT_TEMPLATE.read_text().replace("__REPORT_JSON__", payload)


def rendered_page(playwright, report_md: str, title: str):
    """(browser, page) with the report rendered and its fonts loaded. Caller closes the browser."""
    html = _page_html(report_md, title)
    browser = playwright.chromium.launch(args=["--no-sandbox"])
    try:
        page = browser.new_page()

        def answer(route):
            path = urlparse(route.request.url).path
            if path == "/print":
                return route.fulfill(body=html, content_type="text/html; charset=utf-8")
            local = static_file(path)
            if local is None:
                return route.abort()
            return route.fulfill(path=str(local))

        page.route("**/*", answer)
        page.goto(f"{ORIGIN}/print", wait_until="load", timeout=RENDER_TIMEOUT_MS)
        page.wait_for_function("window.__rendered === true", timeout=RENDER_TIMEOUT_MS)
        page.evaluate("document.fonts.ready.then(() => true)")
        return browser, page
    except Exception:
        browser.close()
        raise


def render_pdf(report_md: str, title: str) -> bytes:
    """The report as a Letter-size PDF with page numbers."""
    from playwright.sync_api import sync_playwright

    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    with sync_playwright() as p:
        browser, page = rendered_page(p, report_md, title)
        try:
            return page.pdf(
                format="Letter",
                print_background=True,
                margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                display_header_footer=True,
                header_template="<span></span>",
                footer_template=FOOTER.format(title=safe_title),
            )
        finally:
            browser.close()
