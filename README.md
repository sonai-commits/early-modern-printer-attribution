# cdt-printer-attribution

A system for identifying who printed an anonymous Early Modern English
book, by analyzing the tiny physical damage marks on the printed letters.

Built around a natural-language agent that can use thirteen specialized
analysis tools, weigh bibliographic evidence against visual evidence,
and explain its reasoning step by step.

---

## The problem

Many books printed in seventeenth-century England carry false or missing
information on their title pages. Printers who handled politically risky
material—unlicensed pamphlets, dissenting religious texts, satirical
attacks on the crown—often left their names off, used a fake imprint,
or hid behind a bookseller. Identifying the real printer of an anonymous
book is a classic problem in bibliography.

One useful clue: every printing shop owned a finite set of metal type
pieces, and over time those individual pieces accumulated small physical
nicks and bends. Two books printed with the same set of damaged pieces
were almost certainly printed in the same shop. The
[Catalog of Distinctive Type (CDT)](https://cdt.library.cmu.edu) at
Carnegie Mellon catalogs thousands of these damaged characters across
hundreds of identified printers, providing a reference set to compare
against.

This project turns that reference set into an interactive attribution
system.

## What this system does

Given an Early Modern book in the CDT corpus:

1. It extracts visual signatures from the damaged characters in the book.
2. It compares those signatures against per-printer fingerprints built
   from the rest of the corpus.
3. It combines this visual evidence with other clues—the date, the
   stated publisher, known bookseller partnerships—using probabilistic
   reasoning.
4. It can adaptively decide which clues to gather first, stopping early
   when an answer becomes clear.
5. The whole pipeline is wrapped in a language-model agent that you
   can talk to in natural English and that explains every step of its
   reasoning.

The system runs on a single GPU machine and offers both a command-line
interface and a web-based chat interface.

## Important limitations

These are documented up front because they shape what the system can
actually do for a user.

**The system can rank known printers; it cannot identify unknown ones.**
On books whose true printer has multiple examples in our corpus, the
correct printer appears in the top 3 ranked candidates about 67% of the
time. On books whose true printer has only one example in the corpus
(an unknown printer in practice), the correct printer appears in the
top 3 zero percent of the time across the 11 such books we tested.

The system detects this situation automatically and warns the user
rather than offering a confident wrong answer. But the underlying point
remains: this is a tool for narrowing down a list of plausible printers,
not for discovering an unknown one.

**Corpus scope is limited.** The current evaluation covers 59 printers
and around 3,800 character images from CDT, focused on English print
between roughly 1645 and 1704. Books outside this period or geography
are not supported without rebuilding the corpus.

**This is not a production deployment.** The system is intended to
support bibliographic research and to demonstrate an integrated
agentic-RAG architecture. It has not been optimized for high traffic
or hardened against adversarial input.

## How it works

### The pipeline, in five layers

**Layer 1: Image preprocessing.** Each character image is normalized,
background-removed, and rendered as a damage residual against a clean
template of the same letter.

**Layer 2: Embedding.** A contrastive convolutional neural network,
trained on the corpus, maps each damaged character to a 128-dimensional
vector. Characters from the same physical type piece end up close
together in this space.

**Layer 3: Clustering.** Characters are grouped into clusters of likely
same-piece matches. The current corpus produces 165 clusters with
about 28% noise (characters that don't fit any cluster).

**Layer 4: Fingerprints.** Each printer gets a TF-IDF-style fingerprint
over the cluster space, capturing which damage patterns appear in their
books and how distinctively.

**Layer 5: Multi-evidence aggregation.** Visual evidence is combined
with three other streams—imprint regex matching, temporal range
overlap, and bookseller partnership likelihood—via Bayesian
likelihood-ratio aggregation into a single posterior probability over
candidate printers.

### The agent

On top of the pipeline sits a Qwen 2.5:14B language model with access
to thirteen tools:

| Tool | Purpose |
|---|---|
| `attribute_book` | Standard vision-only attribution with leave-one-book-out |
| `attribute_book_multi_evidence` | Bayesian aggregation across all four evidence streams |
| `active_attribute_book` | Adaptive stream selection, stops early when confident |
| `compare_printer_fingerprints` | Find typographically similar printers |
| `audit_cluster` | Inspect whether a cluster represents genuine shared damage |
| `find_books_by_printer` | List a printer's catalogued books |
| `search_imprints` | Regex search across publisher fields |
| `lookup_estc` | Fetch the catalog record for a given ESTC identifier |
| `find_similar_glyphs` | k-nearest-neighbor search in glyph embedding space |
| `search_literature` | Semantic search over indexed bibliographic literature |
| `list_printers` | Roster of corpus contents |
| `get_glyph_metadata` | Inspect a specific damaged character |
| `get_empirical_baselines` | Report the encoder's evaluation numbers |

The agent decides which tools to call, in what order, with what
arguments. A two-stage verifier reviews the agent's final answer and
catches fabricated references or unsupported claims. Confidence levels
are computed deterministically in Python and supplied to the agent as
non-negotiable inputs—the agent cannot inflate its own confidence.

### The active acquisition module

Running all four evidence streams costs computational time. When the
imprint clearly names a printer, vision analysis is unnecessary. The
active acquisition module models this explicitly: at each step, it
estimates which unused evidence stream would yield the most information
about the printer, runs that stream, and stops once posterior
probability exceeds a threshold (0.80 by default).

The module supports four masking modes for principled evaluation:

- **Open**: full imprint visible. Routine attribution case.
- **Clandestine**: the true printer's name is masked from the publisher
  field. Simulates a false imprint where the named printer is wrong.
- **Strict-clandestine**: also masks the printer's known bookseller
  partners. Closer to a real anonymous attribution case.
- **Strictest-clandestine**: additionally suppresses the year. Only
  visual evidence carries signal. The worst case.

## Evaluation results

Numbers are recall@3 on 202 books across 59 printers, with chance
performance at 5.5%. The encoder is the contrastive CNN trained for
100 epochs on the full corpus.

| Setting | Recall@3 |
|---|---|
| Standard leave-one-book-out | 63.4% |
| Multi-book printers (subset of 191 books) | 67.0% |
| Singleton-printer books (subset of 11 books — the cold-start test) | **0.0%** |
| Active acquisition, open mode | 99.4% |
| Active acquisition, clandestine | 93.8% |
| Active acquisition, strict-clandestine | 93.8% |
| Active acquisition, strictest-clandestine | 89.3% |

The active acquisition numbers benefit from same-printer cluster
neighbors remaining in the candidate space. A separate diagnostic
called printer-holdout measures how much of the visual similarity is
attributable to those neighbors versus genuine cross-shop damage
matching. The result: when a printer's other books are fully removed
from the candidate space, top-match cosine similarity drops from 0.76
to 0.54. This explains the singleton-subset zero recall.

In plain terms: the visual encoder learned to recognize a printer's
overall typographic character rather than the individual damaged
pieces. Useful for ranking known printers, not for identifying unknown
ones. The repository includes the printer-holdout diagnostic and the
singleton-stratification protocol that produce these numbers, so they
can be applied to other corpora and methods.

## Repository contents

```
cdt-printer-attribution/
├── real_rag.py                  Main pipeline and agent
├── web_chat.py                  Gradio web demo
│
├── scraper.py                   CDT scraping library
├── build_corpus.py              Corpus builder CLI
├── probe_cdt_v6.py              Inventory probe (glyphs per printer)
├── probe_books_per_printer.py   Inventory probe (books per printer)
│
├── diagnose_active.py           Active-acquisition diagnostic
├── diagnose_strict.py           Strict-mode evaluation diagnostic
│
├── results/                     Evaluation result JSONs
├── traces/                      Sample agent reasoning traces
│
├── README.md                    This file
├── requirements.txt             Python dependencies
└── .gitignore
```

The corpus itself (character images and metadata) is not redistributed
here. It belongs to Carnegie Mellon's CDT project. Run `build_corpus.py`
to rebuild it locally from the CDT API. Trained encoder weights are
likewise regenerable.

## Setup

### Requirements

- Python 3.11 or newer
- A CUDA-capable GPU with at least 16 GB of memory (for Qwen 14B
  inference or any other smaller language models can be substituted with edits)
- Around 10 GB of disk space for the corpus, encoder weights, and
  Ollama model

### Step 1: Clone and install Python dependencies

```bash
git clone https://github.com/<your-username>/cdt-printer-attribution.git
cd cdt-printer-attribution
pip install -r requirements.txt
```

### Step 2: Install Ollama and pull the language model

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:14b
```

Ensure the Ollama service is running. The agent connects to
`http://localhost:11434` by default.

### Step 3: Build the corpus

This downloads character images and metadata from the CDT API. The
full build covers 59 printers.

```bash
python build_corpus.py --stage all
```

To run stages individually:

```bash
python build_corpus.py --stage base        # 19 well-represented printers
python build_corpus.py --stage multibook   # 20 additional multi-book printers
python build_corpus.py --stage singletons  # 20 single-book printers
```

The builder is idempotent. Interrupted runs resume from where they
stopped. Add `--rate-limit 1.0` to be more conservative with API calls.

### Step 4: Train the encoder

```bash
python real_rag.py --encoder contrastive --retrain
```

100 epochs (deafult) of contrastive metric learning.

## Usage

### Command-line evaluation

```bash
# Standard recall@3 evaluation with singleton stratification
python real_rag.py --encoder contrastive --evaluate

# Printer-holdout diagnostic
python real_rag.py --encoder contrastive --printer-holdout

# Multi-evidence Bayesian attribution evaluation
python real_rag.py --encoder contrastive --multi-eval

# Active acquisition evaluation (open mode)
python real_rag.py --encoder contrastive --active-eval

# Full active acquisition report across all four masking modes
python real_rag.py --encoder contrastive --active-report
```

### Interactive command-line chat

```bash
python real_rag.py --encoder contrastive --chat
```

The agent maintains conversation context across turns. Slash commands:

| Command | Action |
|---|---|
| `/help` | Show available commands |
| `/reset` | Clear conversation history |
| `/history` | Print conversation so far |
| `/trace` | Print tool calls from the last turn |
| `/save <filename>` | Save the conversation to a file |
| `/baselines` | Show empirical baselines |
| `/quit` | Exit |

### Web interface

```bash
pip install "gradio>=4.0"
python web_chat.py
```

Open `http://127.0.0.1:7860` in a browser. The interface includes 21
example questions covering every tool and every masking mode.

To make the demo accessible publicly via Gradio's tunneling service,
edit `web_chat.py` and set `share=True` in the `demo.launch()` call.
A `https://*.gradio.live` URL appears in the console, valid for 72
hours.

## Example questions

Once the web interface or chat REPL is running, try:

- *Who printed ESTC R175810?*
- *Attribute ESTC R12254 and audit the top contributing cluster.*
- *Use multi-evidence Bayesian attribution on ESTC R175810. Which
  stream contributed most to the ranking?*
- *Run active attribution on ESTC R175810 with strict_clandestine=true
  (mask the printer's name and known booksellers). Compare against
  open mode.*
- *What are the three printers most typographically similar to
  roberts_robert?*
- *Audit cluster A::0006 and tell me whether it represents genuine
  shared damage or single-printer variation.*

For the cold-start scenario:

- *Who printed ESTC R28199?* — this is a book whose printer has only
  one example in the corpus. The system should warn and decline to
  rank confidently.

## Practical guidance for users

If you are using this tool to investigate a real attribution question,
a few suggestions:

1. **Check whether the target printer has multiple examples in the
   corpus before trusting a ranking.** Use `find_books_by_printer` to
   look at the candidate's other books. A printer represented by four
   or more books in the corpus produces meaningful rankings; a printer
   with one or zero books does not.

2. **A leakage_suspect flag at cosine ≥ 0.90 means "there is almost
   certainly a same-printer match in the corpus."** Treat this as
   strong evidence the system found a same-printer book, not as
   independent confirmation that the damaged pieces match.

3. **In active acquisition mode, the cost savings figure tells you
   something about the question's difficulty.** Open-mode attributions
   that save 40%+ of compute are easy cases where the imprint
   essentially gave the answer. Strict-clandestine attributions that
   save 0% are running every available stream because none was decisive
   on its own.

4. **The agent's confidence band is computed in Python, not by the
   language model.** When you see "high confidence," that means the
   underlying cosine similarity gap exceeded a threshold. The agent
   cannot inflate it.

## Related work

- **Print & Probability / CAML** (Vogler, Allen et al., AAAI 2023):
  *Contrastive Attention Networks for Attribution of Early Modern
  Print*. Pairwise damaged-glyph matching trained on synthetic damage
  augmentation, evaluated on hand-curated test pairs from Areopagitica
  and Leviathan Ornaments. Reports Recall@5 of 58.15% on the Leviathan
  hard-negative pool. [arxiv.org/abs/2306.07998](https://arxiv.org/abs/2306.07998)
- **Areopagitica attribution** (Warren et al. 2020): the manual
  bibliographic study that established the methodology for damaged-
  type attribution as currently practiced.
- **Leviathan Ornaments** (Warren et al. 2021): the bibliographic
  resolution of the famously misattributed Hobbes editions, used as
  ground truth in subsequent computational work.

Our system differs in two respects. We operate at the printer level
(ranking 59 candidate printers) rather than the pairwise-match level
(deciding whether two specific glyphs are the same physical piece).
And we combine visual evidence with bibliographic evidence in an
agentic, tool-driven workflow rather than as a fixed pipeline.

## Acknowledgments

This work uses the
[Catalog of Distinctive Type](https://cdt.library.cmu.edu)
from Carnegie Mellon University Libraries. All character images and
catalog metadata are CDT's, accessed through their open API. The
encoder, fingerprint pipeline, agent design, evaluation diagnostics,
and chat interface are this project's contribution.


## License

Code is released under the MIT License. Character images, catalog
metadata, and any artifacts derived from CDT data (including trained
encoder weights) are subject to Carnegie Mellon's terms for the
Catalog of Distinctive Type; consult CDT directly for redistribution.
