"""probe_books_per_printer.py — find CDT printers with many distinct books.

Why this matters: our previous expansion picked printers with many GLYPHS
but didn't check whether those glyphs came from many BOOKS or just a few.
Printers with 100 glyphs concentrated in 2 books cause structural leakage
in leave-one-book-out evaluation: removing one book still leaves the
other book defining nearly the same fingerprint.

This probe:
  1. Hits the same /?printer_like= endpoint as probe_cdt_v6.py.
  2. Extracts the BOOK ID from each glyph's unique_id (e.g. 'R1691' from
     'Roberts_R1691.003').
  3. Counts distinct book IDs per printer.
  4. Reports top candidates for an honest-evaluation expansion.

Run:
    python probe_books_per_printer.py

Outputs:
    books_per_printer.json
    multibook_expansion_20.py    (paste-ready list of 20 printers)

Caveat: the CDT search appears to cap results at 100 glyphs per page.
A printer with 200 glyphs across 10 books may only show 100 glyphs from
~5 visible books here. So "books_visible" is a LOWER BOUND on each
printer's true book count.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

UA = {"User-Agent": "cdt-multibook-probe/1.0 (research)"}
CDT = "https://cdt.library.cmu.edu"
SLEEP = 1.5

# Reuse the printer roster discovered by probe_cdt_v6.py.
# This is the full CDT API list of 170 printers + Lichfield (the one
# that returns zero glyphs but is in our original list).
EXISTING_PRINTERS_19 = {
    "Ibbitson, Robert", "Simmons, Matthew", "Field, John", "Tyler, Evan",
    "Macock, John", "Cole, Peter", "Hayes, John",
    "Newcomb, Thomas", "Roycroft, Thomas", "Flesher, Miles",
    "Hodgkinson, Richard", "Ratcliffe, Thomas", "Cotes, Ellen",
    "Streater, John", "Daniel, Roger", "Roberts, Robert",
    "Roycroft, Samuel", "Bell, Jane", "Hunt, William",
}

# Already-added in the previous round (the 20 we just added).
RECENTLY_ADDED_20 = {
    "Browne, Samuel", "Clowes, John", "Cottrel, James",
    "Best, John", "Cadwell, John", "Dover, Joan",
    "Godbid, William", "Crouch, Edward", "Everingham, Robert",
    "Clark, Mary", "Bennet, Joseph", "Godbid, Ann",
    "Braddyll, Thomas", "Gain, John",
    "Clark, Henry", "Baldwin, Richard", "Astwood, James",
    "Bennet, Margaret", "Hales, Thomas", "Darby, John",
}


def http_get(url, max_bytes=500000):
    """GET with polite delay. Returns body bytes or None on error."""
    time.sleep(SLEEP)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read(max_bytes)
    except Exception as e:
        print(f"    [err] {url}: {e}")
        return None


def fetch_printers_from_api():
    """Get the full printer roster from the CDT API."""
    body = http_get(f"{CDT}/api/printers")
    if not body:
        return []
    try:
        data = json.loads(body)
        return [p["printer_string"] for p in data.get("results", [])
                if "printer_string" in p]
    except Exception as e:
        print(f"  [api-fail] {e}")
        return []


def fetch_glyph_unique_ids(printer_name):
    """Scrape /?printer_like= for the printer. Return their character IDs."""
    queries = [printer_name]
    if "," in printer_name:
        last, first = [s.strip() for s in printer_name.split(",", 1)]
        queries += [last, f"{first} {last}"]
    ids = set()
    for q in queries:
        url = CDT + "/?" + urllib.parse.urlencode({"printer_like": q})
        body = http_get(url)
        if not body:
            continue
        html = body.decode("utf-8", errors="replace")
        new = set(re.findall(r'/characters/([A-Za-z0-9_\.]+)', html))
        ids |= new
        if new:
            break
    return ids


def book_id_from_unique_id(uid):
    """Extract the book key from a unique_id like 'Roberts_R1691.003'.
    The book key is the part between the printer-surname prefix and the
    glyph-number suffix: 'R1691' in this example.

    Pattern: <Surname>_<BookKey>.<NNN>
    Returns the BookKey or None if the format doesn't match."""
    m = re.match(r"^[A-Za-z]+_([A-Za-z]\d+)\.\d+$", uid)
    if m:
        return m.group(1)
    # Fallback: take everything between first underscore and last dot
    m = re.match(r"^[^_]+_(.+)\.\d+$", uid)
    if m:
        return m.group(1)
    return None


def analyze_printer(printer_name):
    """Return (n_glyphs, n_books, book_distribution)."""
    uids = fetch_glyph_unique_ids(printer_name)
    if not uids:
        return 0, 0, {}
    book_ids = []
    unparseable = 0
    for uid in uids:
        bid = book_id_from_unique_id(uid)
        if bid:
            book_ids.append(bid)
        else:
            unparseable += 1
    book_counts = Counter(book_ids)
    return len(uids), len(book_counts), dict(book_counts)


def main():
    print("=" * 72)
    print("BOOKS-PER-PRINTER PROBE")
    print("=" * 72)
    print()
    print("Fetching full printer roster from CDT API...")
    printers = fetch_printers_from_api()
    print(f"  got {len(printers)} printers")
    print()

    # Track which are already in our corpus
    in_corpus = EXISTING_PRINTERS_19 | RECENTLY_ADDED_20

    # Probe each printer for books
    print("=" * 72)
    print(f"Analyzing {len(printers)} printers (estimated "
          f"~{len(printers) * 1.5 / 60:.1f} min)")
    print("=" * 72)
    print()

    results = []
    for i, p in enumerate(printers, 1):
        n_glyphs, n_books, dist = analyze_printer(p)
        already_have = p in in_corpus
        marker = "*" if already_have else " "
        # Books with at least 5 glyphs each — "substantial" books
        substantial = sum(1 for c in dist.values() if c >= 5)
        flag = ""
        if n_books >= 5 and substantial >= 4:
            flag = "  ★ EXCELLENT (≥4 substantial books)"
        elif n_books >= 4:
            flag = "  ◆ GOOD (≥4 books)"
        elif n_books >= 3:
            flag = "  ▪ OK (≥3 books)"
        print(f"  [{i:3d}/{len(printers)}] {marker} {p:38s} "
              f"glyphs={n_glyphs:4d} books={n_books:3d} "
              f"substantial={substantial}{flag}")
        results.append({
            "name": p,
            "n_glyphs": n_glyphs,
            "n_books_visible": n_books,
            "n_substantial_books": substantial,
            "in_corpus": already_have,
            "book_distribution": dist,
        })

    # ─── ANALYSIS ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)

    n_total = len(results)
    n_2plus = sum(1 for r in results if r["n_books_visible"] >= 2)
    n_3plus = sum(1 for r in results if r["n_books_visible"] >= 3)
    n_4plus = sum(1 for r in results if r["n_books_visible"] >= 4)
    n_5plus = sum(1 for r in results if r["n_books_visible"] >= 5)
    n_4subst = sum(1 for r in results if r["n_substantial_books"] >= 4)

    print(f"\n  printers probed:                  {n_total}")
    print(f"  with ≥2 visible books:            {n_2plus}")
    print(f"  with ≥3 visible books:            {n_3plus}")
    print(f"  with ≥4 visible books:            {n_4plus}")
    print(f"  with ≥5 visible books:            {n_5plus}")
    print(f"  with ≥4 substantial books (5+ glyphs each): {n_4subst}")

    # Check our existing corpus
    print(f"\n  YOUR CURRENT 39 PRINTERS:")
    print(f"  {'printer':<30s} books substantial in_corpus")
    print(f"  {'-'*30:<30s}  {'-'*5}  {'-'*11}  {'-'*9}")
    in_corpus_results = [r for r in results if r["in_corpus"]]
    in_corpus_results.sort(key=lambda r: -r["n_books_visible"])
    for r in in_corpus_results:
        print(f"  {r['name']:<30s}  {r['n_books_visible']:5d}  "
              f"{r['n_substantial_books']:11d}  yes")
    n_corpus_4plus = sum(1 for r in in_corpus_results
                         if r["n_books_visible"] >= 4)
    print(f"\n  Of our current 39 printers, {n_corpus_4plus} have ≥4 "
          f"books visible.")

    # Best new candidates: multi-book printers NOT yet in corpus
    new_candidates = [r for r in results if not r["in_corpus"]
                      and r["n_books_visible"] >= 3]
    # Sort by: most substantial books first, then by total books
    new_candidates.sort(key=lambda r: (-r["n_substantial_books"],
                                         -r["n_books_visible"]))

    print()
    print("=" * 72)
    print("TOP 30 NEW CANDIDATES (≥3 books, not yet in corpus)")
    print("  ranked by: substantial books, then total books")
    print("=" * 72)
    print(f"  {'printer':<35s} {'books':>5} {'subst':>5}")
    for r in new_candidates[:30]:
        print(f"  {r['name']:<35s} {r['n_books_visible']:5d} "
              f"{r['n_substantial_books']:5d}")

    # ─── PASTE-READY EXPANSION LIST ──────────────────────────────────────
    # Pick the 20 best multi-book candidates that aren't in corpus yet.
    top_20 = [r["name"] for r in new_candidates[:20]]
    paste_path = Path("multibook_expansion_20.py")
    with paste_path.open("w") as f:
        f.write("# Generated by probe_books_per_printer.py\n")
        f.write("# 20 multi-book printers for honest leave-one-book-out\n")
        f.write("# evaluation. Each has at least 3 distinct books in CDT.\n\n")
        f.write("MULTIBOOK_PRINTERS_20 = [\n")
        for p in top_20:
            r = next((x for x in new_candidates if x["name"] == p), None)
            comment = (f"  # books={r['n_books_visible']}, "
                       f"substantial={r['n_substantial_books']}"
                       if r else "")
            f.write(f'    "{p}",{comment}\n')
        f.write("]\n")

    print(f"\n  paste-ready list saved: {paste_path.resolve()}")

    # Save full inventory
    out_path = Path("books_per_printer.json")
    out_path.write_text(json.dumps({
        "summary": {
            "n_total": n_total,
            "n_2plus_books": n_2plus,
            "n_3plus_books": n_3plus,
            "n_4plus_books": n_4plus,
            "n_5plus_books": n_5plus,
            "n_4_substantial_books": n_4subst,
            "current_corpus_4plus_books": n_corpus_4plus,
        },
        "per_printer": results,
        "top_20_new_multibook": top_20,
    }, indent=2))
    print(f"  full inventory saved: {out_path.resolve()}")

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print()
    if n_4plus < 30:
        print(f"  WARNING: only {n_4plus} CDT printers have ≥4 visible books.")
        print(f"  We can't build a 50-printer corpus where every printer")
        print(f"  has ≥4 books. The best we can do may be a smaller corpus")
        print(f"  with stricter book-count criteria.")
    print()
    print(f"  Your current 39 printers include {n_corpus_4plus} with ≥4 books.")
    print(f"  The other {39 - n_corpus_4plus} have only 1-3 books, which")
    print(f"  causes the structural leakage we observed.")
    print()
    print("  RECOMMENDATION:")
    if n_4plus >= 35:
        print(f"    Replace the {39 - n_corpus_4plus} sparse printers in your")
        print(f"    corpus with multi-book printers from the candidate list.")
        print(f"    This requires another scrape+retrain cycle but should")
        print(f"    produce honest leave-one-book-out evaluation.")
    else:
        print(f"    Add the top 20 candidates and ACCEPT that some printers")
        print(f"    will still be sparse. Stratify the final evaluation by")
        print(f"    books-per-printer to make the honest subset visible.")
    print()
    print("  After expansion: re-scrape (~30-45 min), retrain encoder,")
    print("  rerun --evaluate and --active-report. Numbers will probably")
    print("  drop substantially but be HONEST.")


if __name__ == "__main__":
    main()
