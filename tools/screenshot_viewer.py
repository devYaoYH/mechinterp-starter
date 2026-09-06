"""Regenerate the viewer screenshots in docs/.

Optional tooling, not part of requirements.txt -- you only need it to refresh
the images in the README:

    uv pip install playwright && python -m playwright install chromium
    python tools/screenshot_viewer.py
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)


def shoot(page, theme, name, full=True):
    page.emulate_media(color_scheme=theme)
    page.evaluate("t => document.documentElement.setAttribute('data-theme', t)", theme)
    page.wait_for_timeout(400)
    path = OUT / name
    page.screenshot(path=str(path), full_page=full)
    print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} kB)")


def main():
    src = (ROOT / "viewer.html").resolve()
    if not src.exists():
        sys.exit("viewer.html not found")
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": 1180, "height": 900}, device_scale_factor=2)
        page.goto(src.as_uri())
        page.wait_for_selector("#checks .check")
        page.wait_for_timeout(600)          # webfonts
        shoot(page, "light", "viewer-light.png")
        shoot(page, "dark", "viewer-dark.png")
        b.close()


if __name__ == "__main__":
    main()
