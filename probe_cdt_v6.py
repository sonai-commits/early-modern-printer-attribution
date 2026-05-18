"""probe_cdt_v6.py — comprehensive CDT inventory diagnostic.

Goal: figure out exactly what's available on CDT so we can plan corpus
expansion. Does NOT download images; only inspects metadata.

Phase 1: discover the printer-listing API endpoint (try several patterns).
Phase 2: paginate through every printer in the API.
Phase 3: count glyphs per printer via /?printer_like= scraping.
Phase 4: diagnose specifically why Lichfield, Leonard failed in our
         original scrape (we know it returned 0 IDs).
Phase 5: report new candidate printers with >= 50 glyphs.
Phase 6: emit a paste-ready PRINTERS_EXPANDED list for history_rag.py.

Run:
    python probe_cdt_v6.py
Outputs:
    cdt_inventory_v6.json
    printers_expanded.py
"""
import urllib.request
import urllib.parse
import urllib.error
import json
import time
import re
from pathlib import Path

UA = {"User-Agent": "cdt-corpus-probe/0.6 (research; debanjan)"}
CDT = "https://cdt.library.cmu.edu"
PSC = "https://printprobdb.psc.edu"
SLEEP = 1.5
MAX_PROBES = 250    # ceiling on how many printers we'll glyph-count
GLYPH_THRESHOLDS = [20, 35, 50, 100]    # report counts at each threshold

CURRENT_PRINTERS = [
    "Ibbitson, Robert", "Simmons, Matthew", "Field, John", "Tyler, Evan",
    "Macock, John", "Cole, Peter", "Lichfield, Leonard", "Hayes, John",
    "Newcomb, Thomas", "Roycroft, Thomas", "Flesher, Miles",
    "Hodgkinson, Richard", "Ratcliffe, Thomas", "Cotes, Ellen",
    "Streater, John", "Daniel, Roger", "Roberts, Robert",
    "Roycroft, Samuel", "Bell, Jane", "Hunt, William",
]


def get(url, max_bytes=500000):
    time.sleep(SLEEP)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(max_bytes), None
    except urllib.error.HTTPError as e:
        return e.code, e.read(2000) if e.fp else b"", str(e)
    except Exception as e:
        return None, b"", str(e)


def get_json(url):
    s, body, err = get(url)
    if err:
        return None, err
    if s != 200:
        return None, f"HTTP {s}"
    try:
        return json.loads(body), None
    except Exception as e:
        return None, f"parse error: {e}"


# ─────────────────────────────────────────────────────────────────────────
# PHASE 1: discover the printer-listing endpoint
# ─────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1: discover printer-listing endpoints")
print("=" * 70)

endpoints_to_try = [
    f"{PSC}/api/printers/?format=json&page_size=200",
    f"{PSC}/api/printers/?format=json",
    f"{PSC}/api/printers/",
    f"{CDT}/api/printers",
    f"{CDT}/api/printers/",
    f"{CDT}/api/v1/printers/",
    f"{CDT}/printers/",
    f"{CDT}/printers",
]

printers_data = None
working_endpoint = None
for url in endpoints_to_try:
    print(f"\n  trying: {url}")
    data, err = get_json(url)
    if err:
        print(f"    -> {err}")
        continue
    if isinstance(data, dict):
        if "results" in data:
            n = data.get("count", len(data["results"]))
            print(f"    OK: paginated, count={n}, "
                  f"page_size={len(data['results'])}")
            if data['results']:
                print(f"    first record keys: "
                      f"{list(data['results'][0].keys())}")
            printers_data = data
            working_endpoint = url
            break
        else:
            print(f"    OK: dict, keys={list(data.keys())[:10]}")
    elif isinstance(data, list):
        print(f"    OK: list of {len(data)}")
        if data:
            print(f"    first keys: "
                  f"{list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        printers_data = data
        working_endpoint = url
        break

if not printers_data:
    print("\n  ✗ no printer-listing endpoint worked.")
else:
    print(f"\n  ✓ working endpoint: {working_endpoint}")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 2: enumerate all available printers
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 2: enumerate all available printers")
print("=" * 70)

all_printers = []
if printers_data:
    if isinstance(printers_data, dict) and "results" in printers_data:
        all_printers.extend(printers_data["results"])
        next_url = printers_data.get("next")
        page = 1
        while next_url and page < 50:    # safety cap on pagination
            page += 1
            print(f"  fetching page {page}: {next_url}")
            data, err = get_json(next_url)
            if err:
                print(f"    pagination error: {err}; stopping")
                break
            if isinstance(data, dict) and "results" in data:
                all_printers.extend(data["results"])
                next_url = data.get("next")
            else:
                break
    elif isinstance(printers_data, list):
        all_printers = printers_data

    print(f"\n  total printers in API: {len(all_printers)}")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 3: count glyphs per printer
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 3: glyph counts per printer")
print("=" * 70)


def count_glyphs_for_printer(printer_name):
    """Scrape the CDT HTML for a printer's character IDs.

    Try multiple query variants because the form is fuzzy:
      - 'Last, First'
      - 'Last' alone
      - 'First Last'
    Return both the set of IDs and the list of variants tried (for diagnostic)."""
    queries = [printer_name]
    if "," in printer_name:
        last, first = [s.strip() for s in printer_name.split(",", 1)]
        queries += [last, f"{first} {last}"]
    elif " " in printer_name:
        parts = printer_name.split(" ", 1)
        queries.append(f"{parts[-1]}, {parts[0]}")
    ids = set()
    tried = []
    for q in queries:
        time.sleep(SLEEP)
        url = CDT + "/?" + urllib.parse.urlencode({"printer_like": q})
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
            new_ids = set(re.findall(r'/characters/([A-Za-z0-9_\.]+)', html))
            ids |= new_ids
            tried.append({"query": q, "n_found": len(new_ids)})
            if new_ids:
                break
        except Exception as e:
            tried.append({"query": q, "error": str(e)})
    return ids, tried


# Build the list of printer names to count
names_to_count = []
api_record_by_name = {}
if all_printers:
    name_keys = ["name", "full_name", "printer_name", "label", "display_name"]
    for p in all_printers:
        if isinstance(p, dict):
            for k in name_keys:
                if k in p and p[k]:
                    nm = p[k].strip()
                    names_to_count.append(nm)
                    api_record_by_name[nm] = p
                    break
            else:
                for k, v in p.items():
                    if isinstance(v, str) and 0 < len(v) < 80:
                        nm = v.strip()
                        names_to_count.append(nm)
                        api_record_by_name[nm] = p
                        break
else:
    print("  no API list; falling back to the hardcoded 20.")
    names_to_count = list(CURRENT_PRINTERS)

# Make sure all current printers are also in the probe list (even if not
# in API), so we can re-diagnose Lichfield.
for cp in CURRENT_PRINTERS:
    if cp not in names_to_count:
        names_to_count.append(cp)

names_to_count = names_to_count[:MAX_PROBES]
print(f"  probing {len(names_to_count)} printers (estimated "
      f"~{len(names_to_count) * 3 * SLEEP / 60:.1f} min worst-case)...")
print()

results = []
for i, name in enumerate(names_to_count):
    ids, tried = count_glyphs_for_printer(name)
    n = len(ids)
    bucket = ("≥100" if n >= 100 else
              "50-99" if n >= 50 else
              "20-49" if n >= 20 else
              "10-19" if n >= 10 else
              "<10")
    in_corpus = (name in CURRENT_PRINTERS) or any(
        name.replace(", ", "").lower() == p.replace(", ", "").lower()
        for p in CURRENT_PRINTERS)
    marker = "*" if in_corpus else " "
    print(f"  [{i+1:3d}/{len(names_to_count)}] {marker} {name:38s} "
          f"glyphs={n:4d} ({bucket})")
    results.append({
        "name": name,
        "n_glyphs": n,
        "in_current_corpus": in_corpus,
        "queries_tried": tried,
    })


# ─────────────────────────────────────────────────────────────────────────
# PHASE 4: diagnose Lichfield failure specifically
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 4: diagnose why Lichfield, Leonard fails")
print("=" * 70)

lichfield = next((r for r in results if "lichfield" in r["name"].lower()),
                 None)
if lichfield:
    print(f"\n  Lichfield entry: glyphs={lichfield['n_glyphs']}")
    print(f"  Queries tried:")
    for t in lichfield["queries_tried"]:
        if "error" in t:
            print(f"    '{t['query']}' -> ERROR: {t['error']}")
        else:
            print(f"    '{t['query']}' -> found {t['n_found']} IDs")
    if lichfield["n_glyphs"] == 0:
        print("\n  Lichfield genuinely returns 0 across all name variants.")
        print("  Likely root causes (need manual investigation):")
        print("    1. Lichfield is in API but his character records are not "
              "indexed by CDT's printer_like form filter.")
        print("    2. Lichfield's character IDs use a different URL prefix.")
        print("    3. Lichfield is referenced by alternate name in CDT.")
        print()
        print("  Next diagnostic: try direct PSC API lookup if available")
        # If we have an API record, peek at it
        api_rec = api_record_by_name.get(lichfield["name"])
        if api_rec:
            print(f"  API record for Lichfield:")
            print(f"    {json.dumps(api_rec, indent=4)[:500]}")
else:
    print("  Lichfield, Leonard was not probed.")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 5: inventory summary
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 5: inventory summary")
print("=" * 70)

n_total = len(results)
print(f"\n  probed: {n_total} printers")
print(f"\n  glyph count distribution:")
for threshold in GLYPH_THRESHOLDS:
    n_at = sum(1 for r in results if r["n_glyphs"] >= threshold)
    n_at_new = sum(1 for r in results if r["n_glyphs"] >= threshold
                    and not r["in_current_corpus"])
    print(f"    ≥{threshold:3d} glyphs: {n_at:3d} total, "
          f"{n_at_new:3d} not yet in corpus")

new_candidates = sorted(
    [r for r in results if r["n_glyphs"] >= 20
     and not r["in_current_corpus"]],
    key=lambda r: -r["n_glyphs"])
print(f"\n  TOP 20 NEW CANDIDATES (≥20 glyphs, not in corpus):")
for r in new_candidates[:20]:
    print(f"    {r['name']:38s} glyphs={r['n_glyphs']}")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 6: emit paste-ready PRINTERS_EXPANDED for history_rag.py
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 6: paste-ready expansion list for history_rag.py")
print("=" * 70)

# Tier the expansion. We'll suggest two thresholds:
#   conservative: only printers with >= 50 glyphs
#   aggressive:   include printers with >= 20 glyphs
for label, threshold in [("CONSERVATIVE (≥50 glyphs)", 50),
                          ("AGGRESSIVE (≥20 glyphs)", 20)]:
    expanded = list(CURRENT_PRINTERS)
    added = [r["name"] for r in new_candidates
             if r["n_glyphs"] >= threshold]
    expanded.extend(added)
    print(f"\n  {label}: adds {len(added)} printers, "
          f"total = {len(expanded)}")

# Save the paste-ready file
expanded_conservative = list(CURRENT_PRINTERS) + [
    r["name"] for r in new_candidates if r["n_glyphs"] >= 50]
expanded_aggressive = list(CURRENT_PRINTERS) + [
    r["name"] for r in new_candidates if r["n_glyphs"] >= 20]

paste_path = Path("printers_expanded.py")
with paste_path.open("w") as f:
    f.write("# Generated by probe_cdt_v6.py\n")
    f.write("# Paste either list into history_rag.py to replace PRINTERS.\n")
    f.write("# Then re-run download_cdt_corpus().\n\n")
    f.write(f"# Conservative: {len(expanded_conservative)} printers, "
            f"only those with >= 50 glyphs\n")
    f.write("PRINTERS_CONSERVATIVE = [\n")
    for p in expanded_conservative:
        f.write(f'    "{p}",\n')
    f.write("]\n\n")
    f.write(f"# Aggressive: {len(expanded_aggressive)} printers, "
            f"includes those with >= 20 glyphs\n")
    f.write("PRINTERS_AGGRESSIVE = [\n")
    for p in expanded_aggressive:
        f.write(f'    "{p}",\n')
    f.write("]\n")
print(f"\n  paste-ready expansion list saved: {paste_path.resolve()}")


# Save the full inventory JSON
out_path = Path("cdt_inventory_v6.json")
out_path.write_text(json.dumps({
    "working_endpoint": working_endpoint,
    "total_in_api": len(all_printers) if printers_data else None,
    "probed_count": n_total,
    "summary": {
        "at_threshold_" + str(t): {
            "total": sum(1 for r in results if r["n_glyphs"] >= t),
            "not_in_corpus": sum(1 for r in results if r["n_glyphs"] >= t
                                  and not r["in_current_corpus"]),
        }
        for t in GLYPH_THRESHOLDS
    },
    "per_printer": results,
    "new_candidates_top20": new_candidates[:20],
    "expanded_conservative_size": len(expanded_conservative),
    "expanded_aggressive_size": len(expanded_aggressive),
}, indent=2))
print(f"  full inventory saved: {out_path.resolve()}")
print()
print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("  1. Review TOP 20 NEW CANDIDATES above.")
print("  2. Pick CONSERVATIVE or AGGRESSIVE expansion.")
print("  3. Open printers_expanded.py, copy the chosen list.")
print("  4. Open history_rag.py and replace PRINTERS = [...] with it.")
print("  5. Run: python history_rag.py")
print("     This will re-scrape (~1-2 hours for aggressive list)")
print("  6. Run: python real_rag.py --retrain --encoder contrastive")
print("     (retrain encoder; ~30-60 min)")
print("  7. Run: python real_rag.py --evaluate --encoder contrastive")
print("     (standard recall@3 baseline; ~5 min)")
print("  8. Run: python real_rag.py --active-report --encoder contrastive")
print("     (full active eval, all four masking modes; ~25 min)")
