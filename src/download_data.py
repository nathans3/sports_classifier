"""
Sport Classifier — Data Download Script
=========================================
Downloads basketball player and soccer player images from DuckDuckGo.

Step 1 of the fastai Chapter 2 workflow: gather the data.

Usage::
    cd <project_root>
    python3 src/download_data.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastai.vision.all import get_image_files
from utils import search_images_ddg, download_images_to_folder, clean_dataset, deduplicate_urls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

IMAGES_PER_QUERY = 50
SEARCH_DELAY     = 7.0
CATEGORY_DELAY   = 20.0

SEARCH_QUERIES: dict[str, list[str]] = {
    "basketball": [
        "NBA basketball player dribbling court action",
        "basketball player shooting layup game",
        "professional basketball player jumping dunking",
        "basketball game player running court",
    ],
    "soccer": [
        "soccer player kicking ball field action",
        "football player dribbling pitch game",
        "professional soccer player running stadium",
        "soccer player heading ball match",
    ],
}


def download_category(category: str, queries: list[str]) -> None:
    dest = DATA_DIR / category
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n📁  Category: {category.upper()}")
    print("─" * 44)
    all_urls = []
    for query in queries:
        urls = search_images_ddg(query, max_images=IMAGES_PER_QUERY)
        all_urls.append(urls)
        time.sleep(SEARCH_DELAY)
    unique = deduplicate_urls(all_urls)
    print(f"  Total unique URLs: {len(unique)}")
    download_images_to_folder(dest, unique)


def print_summary() -> None:
    print("\n" + "═" * 44)
    print("  📊  Final Image Counts")
    print("═" * 44)
    total = 0
    for cat in SEARCH_QUERIES:
        n = len(get_image_files(DATA_DIR / cat)) if (DATA_DIR / cat).exists() else 0
        print(f"  {cat:<14}: {n:4d} images")
        total += n
    print("  " + "─" * 22)
    print(f"  TOTAL         : {total:4d} images")
    print("═" * 44)


def main() -> None:
    print("🏀⚽ Sport Classifier — Data Download")
    print("═" * 44)
    print(f"  Categories   : {', '.join(SEARCH_QUERIES)}")
    print(f"  Queries/cat  : {len(next(iter(SEARCH_QUERIES.values())))}")
    print(f"  Images/query : {IMAGES_PER_QUERY}")
    print("═" * 44)

    cats = list(SEARCH_QUERIES.items())
    for i, (cat, queries) in enumerate(cats):
        download_category(cat, queries)
        if i < len(cats) - 1:
            print(f"\n  ⏳ Cooling down {CATEGORY_DELAY:.0f}s before next category…")
            time.sleep(CATEGORY_DELAY)

    print("\n\n🧹  Cleaning Dataset")
    total_removed = 0
    for cat in SEARCH_QUERIES:
        print(f"\n  Cleaning {cat}/…")
        total_removed += clean_dataset(DATA_DIR / cat)
    print(f"\n  Total removed: {total_removed}")
    print_summary()
    print("\n✅  Download complete!  Next: python3 src/train.py")


if __name__ == "__main__":
    main()
