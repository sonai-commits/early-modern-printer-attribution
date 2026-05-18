"""scraper.py — CDT scraping library.

for fetching CDT character IDs, metadata, and images.
Imported by build_corpus.py and the probe
scripts.

Public API:
    fetch_printer_character_ids(printer_name)  -> sorted list of char IDs
    fetch_character_record(char_id)            -> dict (one glyph's metadata)
    download_image(url, out_path)              -> bool (success)
    download_one_printer(printer, out_dir, ...) -> (n_attempted, n_landed)
    pick_single_book_and_download(printer, out_dir, ...) -> dict
    rebuild_manifest(out_dir, all_printers)    -> list of records
"""
import io
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────
CDT_BASE     = "https://cdt.library.cmu.edu"
USER_AGENT   = "cdt-printer-attribution/1.0 (research)"
TIMEOUT      = 30
MIN_IMG_BYTES = 300

# Default polite rate — overridable per call
DEFAULT_RATE_LIMIT = 0.4


# ─── HTTP UTILS ─────────────────────────────────────────────────────────
def _http_get(url, timeout=TIMEOUT, retries=5, backoff=2.0):
    """GET bytes with exponential backoff on transient failures."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, socket.timeout, socket.gaierror) as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise last_err


def _http_get_json(url, timeout=TIMEOUT):
    """GET and JSON-parse."""
    return json.loads(_http_get(url, timeout=timeout).decode(
        "utf-8", errors="replace"))


# ─── CORE SCRAPER FUNCTIONS ─────────────────────────────────────────────
def fetch_printer_character_ids(printer_name):
    """Scrape /?printer_like= for a printer's glyph IDs.

    Tries several name variants (full 'Last, First', last-only,
    'First Last') because CDT's form filter is fuzzy. Returns the
    first variant that produced hits."""
    queries = [printer_name]
    if "," in printer_name:
        last, first = [s.strip() for s in printer_name.split(",", 1)]
        queries += [last, f"{first} {last}"]
    ids = set()
    for q in queries:
        try:
            url = (CDT_BASE + "/?"
                   + urllib.parse.urlencode({"printer_like": q}))
            html = _http_get(url).decode("utf-8", errors="replace")
            new = set(re.findall(r'/characters/([A-Za-z0-9_\.]+)', html))
            ids |= new
            if new:
                break
        except Exception as e:
            print(f"  [search] '{q}' failed: {e}")
    return sorted(ids)


def fetch_character_record(char_id):
    """Hit /api/characters/{id} for the parsed metadata record."""
    return _http_get_json(f"{CDT_BASE}/api/characters/{char_id}")


def download_image(url, out_path, min_bytes=MIN_IMG_BYTES):
    """Download and verify an image. Returns True on success."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _http_get(url)
    except Exception as e:
        print(f"  [dl-fail] {url}: {e}")
        return False
    if len(data) < min_bytes:
        return False
    # Magic-byte sniff
    if not (data[:3] == b"\xff\xd8\xff"
            or data[:8] == b"\x89PNG\r\n\x1a\n"):
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except Exception:
        return False
    out_path.write_bytes(data)
    return True


# ─── MANIFEST HELPERS ───────────────────────────────────────────────────
def safe_slug(name):
    """Normalise a printer name into a filesystem-safe slug.
    'Roberts, Robert' -> 'roberts_robert'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_year(rec):
    """Find a year in a character record, falling back to the unique_id
    pattern (e.g. 'Roberts_R1691.003' -> '1691')."""
    for k in ("pq_year_early", "year"):
        if rec.get(k):
            return str(rec[k])[:4]
    book = rec.get("book") or {}
    for k in ("pq_year_early", "pp_year", "year"):
        if book.get(k):
            return str(book[k])[:4]
    m = re.search(r"_[A-Za-z](\d{4})\.", rec.get("unique_id", ""))
    return m.group(1) if m else ""


def book_id_from_uid(uid):
    """Extract book key from glyph unique_id.
    'Roberts_R1691.003' -> 'R1691'."""
    m = re.match(r"^[A-Za-z]+_([A-Za-z]\d+)\.\d+$", uid)
    if m:
        return m.group(1)
    m = re.match(r"^[^_]+_(.+)\.\d+$", uid)
    if m:
        return m.group(1)
    return None


# ─── DOWNLOAD ORCHESTRATION ─────────────────────────────────────────────
def download_one_printer(printer, out_dir, rate_limit=DEFAULT_RATE_LIMIT):
    """Fetch all glyphs for one printer.

    Idempotent: skips already-saved metadata and images on disk.
    Returns (n_attempted, n_landed)."""
    out_dir = Path(out_dir)
    img_root = out_dir / "images"
    meta_root = out_dir / "meta"

    print(f"\n[download] {printer}")
    try:
        ids = fetch_printer_character_ids(printer)
    except Exception as e:
        print(f"  page fetch failed: {e}")
        return 0, 0
    if not ids:
        print(f"  no character ids found for '{printer}'")
        return 0, 0

    slug = safe_slug(printer)
    pdir = img_root / slug
    pdir.mkdir(parents=True, exist_ok=True)
    mdir = meta_root / slug
    mdir.mkdir(parents=True, exist_ok=True)

    n_attempted = len(ids)
    n_landed = 0

    for cid in tqdm(ids, desc=f"  {slug}", leave=False):
        local = pdir / f"{cid}.jpg"
        meta_local = mdir / f"{cid}.json"

        # Cache or fetch metadata
        rec = None
        if meta_local.exists() and meta_local.stat().st_size > 50:
            try:
                rec = json.loads(meta_local.read_text())
            except Exception:
                rec = None
        if rec is None:
            try:
                rec = fetch_character_record(cid)
                meta_local.write_text(json.dumps(rec))
            except Exception as e:
                print(f"  [api-fail] {cid}: {e}")
                continue
            time.sleep(rate_limit)

        web_url = rec.get("web_url")
        if not web_url:
            continue

        if local.exists() and local.stat().st_size >= MIN_IMG_BYTES:
            n_landed += 1
            continue

        ok = download_image(web_url, local)
        if not ok:
            time.sleep(0.5)
            ok = download_image(web_url, local)
        if ok:
            n_landed += 1
        time.sleep(rate_limit)

    print(f"  {slug}: {n_landed}/{n_attempted} glyphs landed")
    return n_attempted, n_landed


def pick_single_book_and_download(printer, out_dir,
                                    rate_limit=DEFAULT_RATE_LIMIT,
                                    min_glyphs_per_book=10):
    """For one printer: pick the book with the most glyphs and download
    only those glyphs. Used for building singleton-printer test cases.

    Returns dict {n_attempted, n_landed, book_key, book_estc}."""
    print(f"\n[singleton] {printer}")
    try:
        ids = fetch_printer_character_ids(printer)
    except Exception as e:
        print(f"  fetch failed: {e}")
        return {"n_attempted": 0, "n_landed": 0,
                "book_key": None, "book_estc": None}
    if not ids:
        print(f"  no characters found")
        return {"n_attempted": 0, "n_landed": 0,
                "book_key": None, "book_estc": None}

    by_book = {}
    for cid in ids:
        bid = book_id_from_uid(cid)
        if bid:
            by_book.setdefault(bid, []).append(cid)
    if not by_book:
        print(f"  no parseable book IDs in {len(ids)} glyphs")
        return {"n_attempted": 0, "n_landed": 0,
                "book_key": None, "book_estc": None}

    best_book = max(by_book.items(), key=lambda kv: len(kv[1]))
    book_key, book_glyph_ids = best_book

    if len(book_glyph_ids) < min_glyphs_per_book:
        print(f"  best book has only {len(book_glyph_ids)} glyphs "
              f"(< {min_glyphs_per_book} required), skipping")
        return {"n_attempted": 0, "n_landed": 0,
                "book_key": None, "book_estc": None}

    print(f"  picked book {book_key} with {len(book_glyph_ids)} glyphs "
          f"(out of {len(by_book)} books, {len(ids)} total glyphs)")

    out_dir = Path(out_dir)
    slug = safe_slug(printer)
    pdir = out_dir / "images" / slug
    pdir.mkdir(parents=True, exist_ok=True)
    mdir = out_dir / "meta" / slug
    mdir.mkdir(parents=True, exist_ok=True)

    n_landed = 0
    book_estc = None
    for cid in tqdm(book_glyph_ids, desc=f"  {slug}", leave=False):
        local = pdir / f"{cid}.jpg"
        meta_local = mdir / f"{cid}.json"

        rec = None
        if meta_local.exists() and meta_local.stat().st_size > 50:
            try:
                rec = json.loads(meta_local.read_text())
            except Exception:
                rec = None
        if rec is None:
            try:
                rec = fetch_character_record(cid)
                meta_local.write_text(json.dumps(rec))
            except Exception as e:
                print(f"  [api-fail] {cid}: {e}")
                continue
            time.sleep(rate_limit)

        if book_estc is None:
            book_estc = (rec.get("book") or {}).get("estc", "")

        web_url = rec.get("web_url")
        if not web_url:
            continue

        if local.exists() and local.stat().st_size >= MIN_IMG_BYTES:
            n_landed += 1
            continue

        ok = download_image(web_url, local)
        if not ok:
            time.sleep(0.5)
            ok = download_image(web_url, local)
        if ok:
            n_landed += 1
        time.sleep(rate_limit)

    print(f"  {slug}: {n_landed}/{len(book_glyph_ids)} glyphs from "
          f"book {book_key} (ESTC {book_estc})")
    return {"n_attempted": len(book_glyph_ids), "n_landed": n_landed,
            "book_key": book_key, "book_estc": book_estc}


def rebuild_manifest(out_dir, all_printers):
    """Walk disk and rebuild manifest.json from whatever's there.

    The manifest is the canonical record of what's been scraped — used
    by real_rag.py to build the corpus index."""
    out_dir = Path(out_dir)
    img_root = out_dir / "images"
    meta_root = out_dir / "meta"

    records = []
    n_dropped = 0
    for printer in all_printers:
        slug = safe_slug(printer)
        pdir = img_root / slug
        mdir = meta_root / slug
        if not pdir.exists():
            continue
        for jpg in sorted(pdir.glob("*.jpg")):
            cid = jpg.stem
            mfile = mdir / f"{cid}.json"
            if not mfile.exists():
                continue
            try:
                with Image.open(jpg) as im:
                    im.verify()
            except Exception:
                n_dropped += 1
                continue
            try:
                rec = json.loads(mfile.read_text())
            except Exception:
                n_dropped += 1
                continue
            book = rec.get("book") or {}
            records.append({
                "id": cid,
                "printer_slug": slug,
                "char": rec.get("character_class", "?"),
                "year": parse_year(rec),
                "path": str(jpg.relative_to(out_dir)),
                "estc": book.get("estc", ""),
                "title": book.get("pq_title", ""),
                "publisher": (book.get("pp_publisher", "")
                              or book.get("pq_publisher", "")),
                "iiif_url": rec.get("web_url", ""),
            })
    if n_dropped:
        print(f"  [clean] dropped {n_dropped} corrupt records")
    (out_dir / "manifest.json").write_text(
        json.dumps({"printers": all_printers, "records": records}, indent=2))
    return records
