"""build_corpus.py — single entry point for building the CDT corpus.

Replaces the older expand_corpus_20.py and expand_singletons_20.py
scripts. Stages:

  base        — original 19 well-represented printers (~1531 glyphs)
  multibook   — add 20 multi-book printers           (~3531 glyphs)
  singletons  — add 20 singleton-printer books       (~3816 glyphs;
                                                       11 are true
                                                       singletons used
                                                       for cold-start
                                                       diagnostic)
  all         — run all three stages in order

After building, retrain the encoder and rebuild caches:

    python real_rag.py --encoder contrastive --retrain

USAGE
=====

  python build_corpus.py --stage all
  python build_corpus.py --stage base
  python build_corpus.py --stage multibook
  python build_corpus.py --stage singletons

OPTIONS
=======

  --rate-limit FLOAT   Seconds between requests (default 0.4)
  --data-dir PATH      Where to put the corpus (default ./sort_rag_run/data/cdt)

All stages are idempotent — interrupted runs resume from where they
left off. Re-running a stage skips already-downloaded files.

CITATION
========

The data this script downloads belongs to Carnegie Mellon University's
Catalog of Distinctive Type. Please cite CDT if you use this corpus.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import scraper as sc


# ─── PRINTER LISTS ──────────────────────────────────────────────────────
# Stage 1: base — the original 19 well-represented printers (≥50 glyphs).
# Lichfield is omitted because CDT's printer_like filter doesn't index
# his character records.
BASE_PRINTERS_19 = [
    "Ibbitson, Robert", "Simmons, Matthew", "Field, John", "Tyler, Evan",
    "Macock, John", "Cole, Peter", "Hayes, John",
    "Newcomb, Thomas", "Roycroft, Thomas", "Flesher, Miles",
    "Hodgkinson, Richard", "Ratcliffe, Thomas", "Cotes, Ellen",
    "Streater, John", "Daniel, Roger", "Roberts, Robert",
    "Roycroft, Samuel", "Bell, Jane", "Hunt, William",
]

# Stage 2: multibook expansion — 20 more multi-book printers, picked for
# date diversity (1649-1704) and ≥100 glyphs each.
MULTIBOOK_PRINTERS_20 = [
    "Browne, Samuel", "Clowes, John", "Cottrel, James",
    "Best, John", "Cadwell, John", "Dover, Joan",
    "Godbid, William", "Crouch, Edward", "Everingham, Robert",
    "Clark, Mary", "Bennet, Joseph", "Godbid, Ann",
    "Braddyll, Thomas", "Gain, John",
    "Clark, Henry", "Baldwin, Richard", "Astwood, James",
    "Bennet, Margaret", "Hales, Thomas", "Darby, John",
]

# Stage 3: singletons — 20 printers from whom we download ONLY their
# single largest book. These create cold-start test cases for the
# diagnostic: each becomes a singleton-printer book with no other
# corpus examples to anchor against.
SINGLETON_PRINTERS_20 = [
    "Griffin, Sarah", "Mason, Edward", "Heptinstall, John",
    "Jones, Edward", "Holt, Ralph", "Moxon, James", "Horton, William",
    "Larkin, George", "Redmayne, William", "Wilde, John",
    "Walter, Robert", "Coe, Andrew", "Lilliecrap, Peter", "Smith, Samuel",
    "Warren, Thomas", "Sowle, Andrew", "Dawson, Gartrude", "Miller, George",
    "Maxwell, Anne", "Twyn, John",
]


# ─── STAGE FUNCTIONS ────────────────────────────────────────────────────
def stage_base(data_dir, rate_limit):
    print("\n" + "=" * 72)
    print(f"STAGE 1: base ({len(BASE_PRINTERS_19)} printers)")
    print("=" * 72)
    t0 = time.time()
    total_attempted = 0
    total_landed = 0
    for printer in BASE_PRINTERS_19:
        try:
            n_a, n_l = sc.download_one_printer(
                printer, data_dir, rate_limit=rate_limit)
            total_attempted += n_a
            total_landed += n_l
        except KeyboardInterrupt:
            print("\n\n[interrupted] partial download saved. "
                  "Re-run to resume.")
            return False
        except Exception as e:
            print(f"  [error on {printer}] {e}; continuing...")
            continue
    print(f"\n[stage:base] {total_landed}/{total_attempted} glyphs "
          f"landed in {(time.time()-t0)/60:.1f} min")
    return True


def stage_multibook(data_dir, rate_limit):
    print("\n" + "=" * 72)
    print(f"STAGE 2: multibook (+{len(MULTIBOOK_PRINTERS_20)} printers)")
    print("=" * 72)
    t0 = time.time()
    total_attempted = 0
    total_landed = 0
    for printer in MULTIBOOK_PRINTERS_20:
        try:
            n_a, n_l = sc.download_one_printer(
                printer, data_dir, rate_limit=rate_limit)
            total_attempted += n_a
            total_landed += n_l
        except KeyboardInterrupt:
            print("\n\n[interrupted] partial download saved. "
                  "Re-run to resume.")
            return False
        except Exception as e:
            print(f"  [error on {printer}] {e}; continuing...")
            continue
    print(f"\n[stage:multibook] {total_landed}/{total_attempted} glyphs "
          f"landed in {(time.time()-t0)/60:.1f} min")
    return True


def stage_singletons(data_dir, rate_limit):
    print("\n" + "=" * 72)
    print(f"STAGE 3: singletons (+{len(SINGLETON_PRINTERS_20)} books "
          f"from new printers — only the single largest book per printer)")
    print("=" * 72)
    print("  This stage creates the cold-start diagnostic cases. Each new")
    print("  printer contributes exactly one book, so leave-one-book-out")
    print("  fully removes the printer from the fingerprint space.")
    t0 = time.time()
    total_landed = 0
    successful = 0
    for printer in SINGLETON_PRINTERS_20:
        try:
            result = sc.pick_single_book_and_download(
                printer, data_dir, rate_limit=rate_limit)
            total_landed += result["n_landed"]
            if result["n_landed"] > 0:
                successful += 1
        except KeyboardInterrupt:
            print("\n\n[interrupted] partial download saved. "
                  "Re-run to resume.")
            return False
        except Exception as e:
            print(f"  [error on {printer}] {e}; continuing...")
            continue
    print(f"\n[stage:singletons] {successful}/"
          f"{len(SINGLETON_PRINTERS_20)} printers successfully added, "
          f"{total_landed} glyphs in {(time.time()-t0)/60:.1f} min")
    return True


def rebuild_and_report(data_dir, all_printers):
    """Rebuild manifest.json and print a summary of what's on disk."""
    print("\n" + "=" * 72)
    print("REBUILDING MANIFEST")
    print("=" * 72)
    records = sc.rebuild_manifest(data_dir, all_printers)
    print(f"\n[manifest] {len(records)} total glyph records across "
          f"{len(all_printers)} printers")

    by_printer = Counter(r["printer_slug"] for r in records)
    by_printer_books = {}
    for r in records:
        by_printer_books.setdefault(r["printer_slug"], set()).add(r["estc"])

    n_singleton = sum(1 for slug, books in by_printer_books.items()
                       if len(books) == 1)
    n_multi = sum(1 for slug, books in by_printer_books.items()
                  if len(books) >= 2)

    print(f"\n  PRINTER BOOK COUNTS:")
    print(f"    singleton-printer (1 book): {n_singleton}")
    print(f"    multi-book printer (≥2 books): {n_multi}")
    print(f"    total active printers in corpus: "
          f"{sum(1 for n in by_printer.values() if n > 0)}")


def invalidate_caches():
    """Wipe stale encoder weights, embeddings, and lancedb."""
    print("\n" + "=" * 72)
    print("CACHE INVALIDATION")
    print("=" * 72)
    runs_dir = Path("./sort_rag_run/runs/cdt_contrastive")
    cache_files = [
        runs_dir / "glyph_embeddings.npy",
        runs_dir / "encoder.pt",
    ]
    for f in cache_files:
        if f.exists():
            print(f"  removing stale cache: {f}")
            f.unlink()
    lancedb_dir = Path("./sort_rag_run/lancedb/contrastive")
    if lancedb_dir.exists():
        print(f"  removing stale lancedb: {lancedb_dir}")
        shutil.rmtree(lancedb_dir)
    print("  done. Next real_rag.py run will retrain and re-embed.")


# ─── CLI ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Build the CDT corpus in stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1] if "USAGE" in __doc__ else "")
    p.add_argument("--stage",
                    choices=["base", "multibook", "singletons", "all"],
                    required=True,
                    help="Which stage(s) to run.")
    p.add_argument("--rate-limit", type=float, default=0.4,
                    help="Seconds between API requests (default 0.4).")
    p.add_argument("--data-dir", type=Path,
                    default=Path("./sort_rag_run/data/cdt"),
                    help="Directory to store the corpus.")
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "images").mkdir(exist_ok=True)
    (args.data_dir / "meta").mkdir(exist_ok=True)

    print("=" * 72)
    print("CDT CORPUS BUILDER")
    print("=" * 72)
    print(f"  data dir:   {args.data_dir}")
    print(f"  rate limit: {args.rate_limit}s")
    print(f"  stage:      {args.stage}")

    if args.stage in ("base", "all"):
        if not stage_base(args.data_dir, args.rate_limit):
            return
    if args.stage in ("multibook", "all"):
        if not stage_multibook(args.data_dir, args.rate_limit):
            return
    if args.stage in ("singletons", "all"):
        if not stage_singletons(args.data_dir, args.rate_limit):
            return

    # Always rebuild manifest with the appropriate printer list
    if args.stage == "base":
        printers = BASE_PRINTERS_19
    elif args.stage == "multibook":
        printers = BASE_PRINTERS_19 + MULTIBOOK_PRINTERS_20
    else:
        printers = (BASE_PRINTERS_19 + MULTIBOOK_PRINTERS_20
                    + SINGLETON_PRINTERS_20)
    rebuild_and_report(args.data_dir, printers)

    invalidate_caches()

    print("\n" + "=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    print()
    print("  python real_rag.py --encoder contrastive --retrain")
    print("  python real_rag.py --encoder contrastive --evaluate")
    print("  python real_rag.py --encoder contrastive --active-report")
    print()


if __name__ == "__main__":
    main()
