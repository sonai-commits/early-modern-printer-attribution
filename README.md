# cdt-printer-attribution

**An agentic RAG framework for damaged-sort printer attribution on the
CMU Catalog of Distinctive Type, with rigorous diagnostics for
distinguishing house-style anchoring from genuine cross-shop damaged-sort
identification.**

---

## What this is

A research-grade Python pipeline that takes an anonymously printed
Early Modern English book and tries to identify which printer actually
produced it, by:

1. Extracting damaged-character glyph signatures from images,
2. Comparing them against a per-printer fingerprint built from the CDT
   corpus,
3. Combining the visual evidence with bibliographic context (imprint,
   date, bookseller partners) via Bayesian likelihood-ratio aggregation,
4. Adaptively choosing which evidence streams to gather via expected
   information gain (active acquisition), and
5. Wrapping all of the above in an LLM-driven agent that can be queried
   in natural language.

It runs end-to-end on a single GPU machine and exposes both a CLI and a
Gradio web interface.

## What this is NOT

This README puts the limitations near the top deliberately, because the
diagnostic findings are part of the contribution.

1. **Not a clandestine attribution solver.** When we evaluate on books
   whose catalogued printer has only one example in the corpus
   ("cold-start" books, n=11), our recall@3 drops to **0.0%**. The
   framework's apparent 67-88% recall on multi-book printers comes
   substantially from house-style anchoring — same-printer cluster
   neighbours in the cluster space — rather than from independent
   damaged-sort identification. We document this finding rigorously.

2. **Not a general-purpose printer ID system.** Scope is 59 printers
   and ~3,800 glyphs from the well-represented subset of CDT
   (~1645–1704 English print). Other corpora are not supported without
   custom data ingestion.

3. **Not a head-to-head improvement over Vogler et al. 2023's CAML
   model.** Their evaluation uses synthetic-trained models on hand-
   curated real test pairs (Areopagitica, Leviathan Ornaments) with
   constructed hard-negative pools. Ours uses leave-one-book-out on
   CDT with structural same-printer leakage. The two numbers are not
   directly comparable.

4. **Not production-deployed.** The active-acquisition and Bayesian
   aggregation pieces are research-grade demonstrations, not optimised
   for serving traffic.

## Contributions, honestly stated

What the framework does contribute:

- **An agentic RAG architecture with 13 specialised tools** for
  bibliographic and typographic analysis, calibrated confidence
  injection, and a two-stage verifier that catches fabricated entity
  references and unsupported tool claims.
- **A Bayesian multi-evidence integration module** that combines vision,
  imprint, temporal, and bookseller streams via log-likelihood-ratio
  aggregation into a single calibrated posterior over candidate printers.
- **An active evidence acquisition module** that adaptively selects
  which evidence stream to run next based on expected information gain
  per unit cost, with four masking modes (open / clandestine /
  strict-clandestine / strictest-clandestine) for principled honest
  evaluation.
- **A printer-holdout and singleton-stratification diagnostic protocol**
  that surfaces house-style anchoring vs damaged-sort identification.
  This is the central scientific result: standard hold-out protocols
  in this domain may overstate true attribution accuracy by tens of
  percentage points, and we provide the tools to measure that gap.
- **A cold-start guardrail** that detects when a book's catalogued
  printer has no other corpus examples and refuses to confidently
  attribute, instead reporting the structural limit honestly.

## Empirical results

Numbers below are from leave-one-book-out evaluation on 202 books
across 59 printers (chance@3 = 5.5%), using a contrastive CNN encoder
(embedding dim 128) trained on all 3,816 glyphs for 100 epochs.

| Regime                                                       | Recall@3 |
|--------------------------------------------------------------|----------|
| Standard (overall, with leakage-flagged books excluded)      | **63.4%** |
| Multi-book-printer subset                                    | 67.0%    |
| **Singleton-printer subset (the honest cold-start test)**    | **0.0%** |
| Active acquisition, open mode (publisher visible)            | 99.4%    |
| Active acquisition, clandestine (printer name masked)        | 93.8%    |
| Active acquisition, strict-clandestine (+ bookseller masked) | 93.8%    |
| Active acquisition, strictest-clandestine (+ year suppressed)| 89.3%    |

The headline finding is the singleton-subset 0.0%. Across 11 books from
11 distinct printers, none of which the encoder has other examples of,
the catalogued printer ranks in the top 3 *zero times*. Printer-holdout
cosine analysis confirms: when a printer's other books are removed from
the cluster space, the top-match cosine drops from 0.76 to 0.54 —
barely above what random pairs of glyphs produce.

Conclusion: this framework rank-recognises known printers but cannot
discover unknown ones. For bibliographers using this kind of tool, the
right use case is hypothesis refinement among a candidate set of known
printers — not blind discovery.

## Repository contents

```
cdt-printer-attribution/
├── real_rag.py                  Main pipeline & agent (~4400 lines)
├── web_chat.py                  Gradio web demo (chat-style interface)
│
├── scraper.py                   CDT scraping library (pure functions)
├── build_corpus.py              Single-entry-point corpus builder
├── probe_cdt_v6.py              CDT inventory discovery (glyphs/printer)
├── probe_books_per_printer.py   Distinct-books-per-printer probe
│
├── diagnose_active.py           Diagnostic: why does active eval fail?
├── diagnose_strict.py           Diagnostic: where does strict recall come from?
│
├── history_rag.py               Legacy: original DINOv2 baseline pipeline,
│                                kept for the DINOv2-vs-contrastive
│                                ablation in the writeup. Superseded
│                                operationally by real_rag.py.
│
├── results/                     Saved evaluation JSONs (recall, ablations)
├── traces/                      Sample agent reasoning traces
│
├── README.md                    This file
├── requirements.txt             Python dependencies
└── .gitignore
```

CDT image data and meta files are **not redistributed** in this repo.
They belong to Carnegie Mellon's CDT project. Run `build_corpus.py` to
rebuild the corpus locally (see Setup below). Trained encoder weights
are also regenerable from the scraped data via `--retrain`.

## Setup

### Requirements

- Python 3.11+ (tested on 3.13)
- CUDA-capable GPU (for encoder training and Qwen inference)
- ~16 GB GPU memory for Qwen 2.5:14b (smaller models work with edits)
- ~10 GB disk space (corpus + cached embeddings + Ollama model)

### Install Python dependencies

```bash
git clone https://github.com/<you>/cdt-printer-attribution.git
cd cdt-printer-attribution
pip install -r requirements.txt
```

### Install Ollama and pull Qwen

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:14b
```

Make sure `ollama serve` is running. The agent connects to
`localhost:11434` by default.

### Build the corpus

This downloads CDT images and metadata for the 39+20 = 59 printers we
evaluated. ~45 minutes of polite-rate scraping. The build is staged
so you can stop after any stage if disk space or time is limited.

```bash
# Build everything in one command (recommended):
python build_corpus.py --stage all

# Or run stages individually:
python build_corpus.py --stage base       # 19 well-represented printers
python build_corpus.py --stage multibook  # +20 multi-book printers
python build_corpus.py --stage singletons # +20 singleton-book printers
                                          # (cold-start diagnostic)
```

The builder is idempotent — interrupted runs resume from where they
left off. Adjust scraping speed with `--rate-limit` (default 0.4s
between requests).

### Train the contrastive encoder

```bash
python real_rag.py --encoder contrastive --retrain
```

100 epochs of contrastive metric learning on the scraped glyphs. ~5–10
minutes on a recent GPU.

## Usage

### CLI: standard evaluation

```bash
# Standard leave-one-book-out recall@3, stratified by singleton vs
# multi-book printers
python real_rag.py --encoder contrastive --evaluate

# Diagnostic: printer-holdout cosine analysis
python real_rag.py --encoder contrastive --printer-holdout

# Module A: multi-evidence Bayesian attribution
python real_rag.py --encoder contrastive --multi-eval

# Module B: active evidence acquisition (open mode)
python real_rag.py --encoder contrastive --active-eval

# Full active report (all 4 masking modes) — the key publishable result
python real_rag.py --encoder contrastive --active-report
```

### CLI: interactive agent

```bash
python real_rag.py --encoder contrastive --chat
```

Multi-turn REPL with conversation memory. Slash commands: `/help /reset
/history /trace /save /baselines /quit`.

### Web interface

```bash
pip install "gradio>=4.0"
python web_chat.py
```

Open `http://127.0.0.1:7860`. The interface ships with 21 suggested
questions covering every tool and every masking mode.

To expose the demo publicly via Gradio's tunneling service, edit
`web_chat.py` and set `share=True` in the `demo.launch()` call. A
`https://*.gradio.live` URL prints to the console; valid for 72 hours.

## How the agent thinks

The agent (Qwen 2.5:14b via Ollama) has access to 13 tools:

| Tool                            | What it does                                       |
|---------------------------------|----------------------------------------------------|
| `attribute_book`                | Vision-only attribution, held-out evaluation       |
| `attribute_book_multi_evidence` | Bayesian aggregation across 4 evidence streams     |
| `active_attribute_book`         | Adaptive stream selection via expected info gain   |
| `compare_printer_fingerprints`  | Cross-printer cluster overlap                       |
| `audit_cluster`                 | Inspect cluster composition for shared-sort genuineness |
| `find_books_by_printer`         | List a printer's catalogued books                  |
| `search_imprints`               | Regex search over publisher fields                 |
| `lookup_estc`                   | Get the metadata record for an ESTC ID             |
| `find_similar_glyphs`           | k-NN over the glyph embedding space (LanceDB)      |
| `search_literature`             | Semantic search over 11 indexed bibliographical works |
| `list_printers`                 | Roster the corpus contents                         |
| `get_glyph_metadata`            | Lookup a specific damaged glyph by ID              |
| `get_empirical_baselines`       | Report the encoder's standard recall@3 baseline   |

The agent decides at each step which to invoke, executes, reads the
result, and either calls another tool or produces a final answer.
A two-stage verifier checks the final answer for fabricated entity
references and unsupported tool claims, with a retry cap of 2 to
prevent infinite re-asking.

Confidence bands are computed deterministically in Python from cosine
similarity and gap-to-second, and **injected as fact the agent cannot
override**. The cold-start warning flag is similarly authoritative.

## Honest caveats for users

If you are a working bibliographer thinking of using this for real
attribution work:

1. The system **rank-recognises printers it has seen multiple examples
   of**. If your target book's printer has 4+ books represented in CDT
   and our corpus, you'll get a useful ranking and an honest confidence
   band.

2. The system **fails on genuinely unknown printers**. If you suspect
   the printer is someone CDT has only one (or zero) examples of, the
   ranking is meaningless. The cold-start guardrail will tell you this.

3. The leakage_suspect flag fires at cosine ≥ 0.90 and indicates that
   other books by the printer probably remained in cluster space.
   Treat as a "very strong same-printer match exists" signal, not as
   independent damaged-sort confirmation.

4. The active acquisition module's cost-savings claim (45% saved in
   open mode) is real but applies when bibliographic imprint context
   is reliable. In strict-clandestine mode the cost-saving drops to
   ~0% because no early-stopping evidence is decisive.

5. We have NOT compared head-to-head against Print & Probability
   (Vogler et al. 2023, AAAI). Their evaluation protocol uses
   synthetic-trained models on different test data. Going to their
   protocol would require a separate effort.

## Related work

- **Print & Probability / CAML** (Vogler, Allen et al., AAAI 2023):
  "Contrastive Attention Networks for Attribution of Early Modern
  Print." Pairwise glyph matching trained on synthetic damage
  augmentation. Recall@5 = 58.15% on Leviathan Ornaments hard-negative
  pool. https://arxiv.org/abs/2306.07998
- **Areopagitica attribution** (Warren et al. 2020): manual
  bibliographical attribution methodology.
- **Leviathan Ornaments** (Warren et al. 2021): definitive Richardson
  attribution; used as ground truth in CAML evaluation.
- **LEAD** (Nov 2025): LLM-enhanced author-name disambiguation hybrid;
  closest analogue to our Module A in a different problem domain.

## Acknowledgments

This work uses the **Catalog of Distinctive Type** from Carnegie Mellon
University Libraries. All glyph images and metadata are CDT's; we
gratefully use their open API. The encoder, fingerprint pipeline,
agent, and diagnostics are this project's contribution.

## Citation

If this work or its diagnostic protocols are useful to you:

```bibtex
@misc{cdt-printer-attribution,
  author = {Debanjan},
  title  = {cdt-printer-attribution: An agentic RAG framework for
            damaged-sort attribution with cold-start diagnostics},
  year   = {2026},
  url    = {https://github.com/<you>/cdt-printer-attribution}
}
```

## License

Code released under the MIT License. CDT data and derived artifacts
(glyph images, encoder weights) are subject to Carnegie Mellon's terms
for the Catalog of Distinctive Type; please consult CDT directly for
redistribution.
