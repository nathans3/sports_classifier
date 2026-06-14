"""
Sport Classifier — Utility Functions
======================================
Reusable helpers for image searching, downloading, and dataset cleaning.
Same structure as the fastai Chapter 2 Bear Classifier utilities.
"""
import time, hashlib, random
from pathlib import Path
import requests
from ddgs import DDGS
from fastai.vision.all import verify_images, get_image_files


def search_images_ddg(term: str, max_images: int = 100) -> list[str]:
    """Search DuckDuckGo for images; return direct image URLs.
    Retries with exponential backoff on rate-limit errors (403/202).
    """
    print(f"  🔍 Searching: '{term}'")
    for attempt in range(4):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(term, max_results=max_images))
            urls = [r["image"] for r in results if r.get("image")]
            print(f"     ✓ {len(urls)} URLs found")
            return urls
        except Exception as exc:
            if attempt < 3:
                wait = 8.0 * (2 ** attempt) + random.uniform(1, 5)
                print(f"     ⏳ Rate limited ({attempt+1}/4), retrying in {wait:.0f}s…")
                time.sleep(wait)
            else:
                print(f"     ✗ Failed after 4 attempts: {exc}")
                return []
    return []


def _download_single(url: str, dest: Path, timeout: int = 10) -> bool:
    try:
        headers = {"User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )}
        resp = requests.get(url, timeout=timeout, headers=headers, stream=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        if "image" not in ct and "octet-stream" not in ct:
            return False
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(8192):
                if chunk:
                    fh.write(chunk)
        return True
    except Exception:
        return False


def download_images_to_folder(dest: Path, urls: list[str], delay: float = 0.05) -> int:
    """Download URLs to dest folder, skipping already-downloaded files."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    success = skipped = failed = 0
    for i, url in enumerate(urls):
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = url.split("?")[0].rstrip("/").rsplit(".", 1)[-1].lower()
        if ext not in {"jpg","jpeg","png","webp","gif"}:
            ext = "jpg"
        fn = dest / f"img_{i:04d}_{url_hash}.{ext}"
        if fn.exists() and fn.stat().st_size > 0:
            skipped += 1
            continue
        if _download_single(url, fn):
            success += 1
        else:
            failed += 1
            if fn.exists():
                fn.unlink()
        if delay > 0 and i < len(urls) - 1:
            time.sleep(delay)
    print(f"     ✓ Downloaded: {success} | Skipped: {skipped} | Failed: {failed}")
    return success


def clean_dataset(folder: Path, verbose: bool = True) -> int:
    """Delete corrupt/unreadable images using fastai verify_images()."""
    folder = Path(folder)
    if not folder.exists():
        if verbose: print(f"  ⚠️  Folder not found: {folder}")
        return 0
    files = get_image_files(folder)
    if not files:
        if verbose: print(f"  No images in {folder.name}/")
        return 0
    if verbose: print(f"  Checking {len(files)} images in {folder.name}/…")
    failed = verify_images(files)
    failed.map(Path.unlink)
    if verbose:
        print(f"  ✗ Removed {len(failed)} corrupt images" if failed else f"  ✓ All {len(files)} images valid")
    return len(failed)


def deduplicate_urls(url_lists: list[list[str]]) -> list[str]:
    """Merge multiple URL lists, removing duplicates while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for lst in url_lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                unique.append(url)
    total_in = sum(len(l) for l in url_lists)
    dupes = total_in - len(unique)
    if dupes:
        print(f"  Removed {dupes} duplicate URLs ({total_in} → {len(unique)})")
    return unique
