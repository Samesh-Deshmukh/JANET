#!/usr/bin/env python3
"""Measure the addressing gate: was this utterance aimed at JANET, or not?

This is the harness that decides whether an NLI model is actually better than
the keyword scorer it would replace. Nothing ships on a hunch — the heuristic in
`intent/scorer.py` is the baseline, and a candidate has to beat it on the same
labelled set.

Run from the repo root:

    ./venv/bin/python tools/addressing_eval.py                 # baseline + default models
    ./venv/bin/python tools/addressing_eval.py --baseline-only # no downloads
    ./venv/bin/python tools/addressing_eval.py --models a,b    # specific checkpoints

The set is `data/text/addressing/eval.jsonl`: one JSON object per line with the
utterance, the conversation turns before it, and a hand-assigned `addressed`
label. Most of it is real speech pulled out of `data/transcripts/`, including
the ambient chatter that leaked through the gates and got answered.

WHY THE TWO ERROR KINDS ARE NOT EQUAL
------------------------------------
A false negative is JANET ignoring you — annoying, you repeat yourself. A false
positive is JANET talking over your conversation, or acting on a sentence from
the television. For a wake-word-free assistant the second is far worse, so the
threshold sweep below reports precision separately rather than hiding both
inside one accuracy number.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EVAL_PATH = REPO / "data" / "text" / "addressing" / "eval.jsonl"

# How many previous exchanges the model is allowed to see. The request was
# "keeping the last 5 turns", and one turn here means one user+JANET pair.
MAX_TURNS = 5

# Candidates worth measuring, smallest first. All of them ship a fast
# `tokenizer.json`, so none needs the `sentencepiece` package.
DEFAULT_MODELS = [
    "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",   # ~71M
    "MoritzLaurer/roberta-base-zeroshot-v2.0-c",             # ~125M
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",            # ~184M
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",           # ~435M
]

# The hypothesis wording is a real tuning knob, not a detail: an NLI model is
# only ever asked "does this hypothesis follow from this premise?", so the
# sentence we write IS the question being asked. Each entry contrasts a
# for-JANET hypothesis against a not-for-JANET one, and the score is a softmax
# over just those two entailment logits — that calibrates far better than
# reading one hypothesis' entailment probability on its own.
HYPOTHESES = {
    "assistant": (
        "The last speaker is talking to the voice assistant.",
        "The last speaker is talking to another person.",
    ),
    "request": (
        "The last speaker is asking the assistant to do something or to answer something.",
        "The last speaker is not asking the assistant for anything.",
    ),
    "janet": (
        "The last line is addressed to Janet, the voice assistant in the room.",
        "The last line is addressed to someone else, or comes from a television or video.",
    ),
}


def load_eval(path=EVAL_PATH):
    """Read the labelled set. Returns a list of dicts."""
    if not path.exists():
        sys.exit(f"eval set not found at {path}")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{n}: bad JSON — {exc}")
    return rows


def load_extra_negatives(repo=REPO):
    """The intent dataset's held-out NONE class, reused as ambient speech.

    `data/text/val/NONE.txt` is 129 lines of exactly what this gate has to
    reject — people talking to each other, television, phone-call fragments,
    and statements that merely mention an alarm or a calendar. It is the
    VALIDATION split, so the DistilBERT classifier was never trained on it.

    One caveat worth stating rather than hiding: these lines follow the intent
    dataset's format contract, so they are lowercase and stripped of
    punctuation, while the rest of the eval set is raw Whisper output. That
    makes them slightly HARDER for an NLI model (which reads natural text
    better) and no harder at all for the keyword scorer, which normalizes
    anyway. The bias therefore runs against the model we are considering
    adopting, which is the safe direction for it to run in.
    """
    path = repo / "data" / "text" / "val" / "NONE.txt"
    if not path.exists():
        return []
    rows = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        rows.append({"id": f"none{n:03d}", "tag": "none_val", "context": [],
                     "utterance": s, "addressed": False,
                     "note": "held-out NONE class from the intent dataset"})
    return rows


def render(row, max_turns=MAX_TURNS):
    """Turn one row into the premise text an NLI model sees.

    The conversation is written as a script because that is the closest thing to
    natural text these models were trained on — a bare concatenation of
    utterances gives them no way to tell who said what, which is the entire
    question being asked here.
    """
    lines = []
    for user, janet in (row.get("context") or [])[-max_turns:]:
        lines.append(f"Person: {user}")
        lines.append(f"Assistant: {janet}")
    lines.append(f"Person: {row['utterance']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Scoring and reporting
# --------------------------------------------------------------------------

def metrics(rows, predictions):
    """Precision/recall/F1 for the ADDRESSED class, plus accuracy.

    Precision is the one that matters most: of everything JANET decided to
    answer, how much of it was actually meant for her?
    """
    tp = sum(1 for r, p in zip(rows, predictions) if p and r["addressed"])
    fp = sum(1 for r, p in zip(rows, predictions) if p and not r["addressed"])
    fn = sum(1 for r, p in zip(rows, predictions) if not p and r["addressed"])
    tn = sum(1 for r, p in zip(rows, predictions) if not p and not r["addressed"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": accuracy}


def report(title, rows, predictions, extra=""):
    m = metrics(rows, predictions)
    print(f"\n  {title}{('  ' + extra) if extra else ''}")
    print(f"    accuracy {m['accuracy']:.1%}   precision {m['precision']:.1%}   "
          f"recall {m['recall']:.1%}   F1 {m['f1']:.3f}")
    print(f"    false positives (answered ambient speech): {m['fp']}   "
          f"false negatives (ignored a real request): {m['fn']}")
    return m


def failures(rows, predictions, limit=12):
    """The rows the gate got wrong, worst kind first."""
    out = []
    for r, p in zip(rows, predictions):
        if p and not r["addressed"]:
            out.append(("FALSE POSITIVE", r))
    for r, p in zip(rows, predictions):
        if not p and r["addressed"]:
            out.append(("false negative", r))
    return out[:limit]


def per_tag(rows, predictions):
    """Accuracy broken down by category, so a weakness has nowhere to hide."""
    tags = {}
    for r, p in zip(rows, predictions):
        t = tags.setdefault(r["tag"], [0, 0])
        t[1] += 1
        if bool(p) == bool(r["addressed"]):
            t[0] += 1
    return tags


# --------------------------------------------------------------------------
# Baseline: the keyword scorer that is currently in the pipeline
# --------------------------------------------------------------------------

def run_baseline(rows):
    """Score every row with intent/scorer.py exactly as dispatch.py does.

    dispatch normalizes before scoring, so we do too — the scorer's word
    matching only fires on the lowercase, punctuation-free form.
    """
    from intent.normalize import normalize
    from intent.scorer import score, THRESHOLD

    print("\n" + "=" * 72)
    print(f"BASELINE — intent/scorer.py keyword heuristic (threshold {THRESHOLD})")
    print("=" * 72)
    print("  Context-blind by construction: it only ever sees the one sentence,")
    print("  which is why the follow-up rows below are the interesting ones.")

    scores, latencies = [], []
    for row in rows:
        t0 = time.perf_counter()
        total, _breakdown = score(normalize(row["utterance"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        scores.append(total)
    predictions = [s >= THRESHOLD for s in scores]
    m = report("at the shipped threshold", rows, predictions,
               f"[{statistics.mean(latencies):.3f} ms/utterance]")
    print("\n    by category:")
    for tag, (ok, n) in sorted(per_tag(rows, predictions).items()):
        print(f"      {tag:<16} {ok}/{n}")
    print("\n    what it gets wrong:")
    for kind, row in failures(rows, predictions):
        print(f"      {kind:<15} {row['id']}  {row['utterance'][:70]!r}")
    return m, scores


def run_live(rows):
    """Score every row through the REAL gate that ships, not a copy of it.

    The benchmark path below builds its own premise and its own hypothesis pair
    so it can compare checkpoints. That makes it a second implementation of the
    thing being measured, and a second implementation is a thing that drifts.
    This mode calls `intent.nli_scorer` directly, so the number printed here is
    the number JANET actually behaves with.
    """
    from intent.nli_scorer import score, THRESHOLD, MODEL_NAME

    class _FakeHistory:
        """Just enough of ConversationHistory for the gate: messages()."""

        def __init__(self, context):
            self._context = context

        def messages(self):
            out = []
            for user, janet in self._context:
                out.append({"role": "user", "content": user})
                out.append({"role": "assistant", "content": janet})
            return out

        def last_reply_was_question(self):
            # The gate reads the conversation ONLY when this is true, so the
            # fake has to answer it or --live measures a different thing.
            return bool(self._context) and self._context[-1][1].strip().endswith("?")

    print("\n" + "=" * 72)
    print(f"LIVE — intent/nli_scorer.py as shipped ({MODEL_NAME})")
    print("=" * 72)

    predictions, latencies = [], []
    for row in rows:
        history = _FakeHistory(row.get("context") or [])
        t0 = time.perf_counter()
        total, _breakdown = score(row["utterance"], history)
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(total >= THRESHOLD)
    m = report("as shipped", rows, predictions,
               f"[{statistics.mean(latencies):.0f} ms mean]")
    print("    " + "  ".join(f"{t} {ok}/{n}" for t, (ok, n)
                             in sorted(per_tag(rows, predictions).items())))
    for kind, row in failures(rows, predictions, limit=12):
        print(f"      {kind:<15} {row['id']}  {row['utterance'][:64]!r}")
    return m


# --------------------------------------------------------------------------
# Candidate: a Hugging Face NLI model used as a zero-shot addressing gate
# --------------------------------------------------------------------------

def entailment_index(config):
    """Which output slot means 'entailment' for this checkpoint.

    MNLI models do NOT agree on label order — some are
    contradiction/neutral/entailment, some the reverse, and a couple of the
    zero-shot checkpoints ship only two classes. Reading it from the config is
    the difference between a working gate and a perfectly inverted one, so it is
    never hard-coded.
    """
    for label, idx in config.label2id.items():
        if "entail" in label.lower():
            return int(idx)
    raise ValueError(f"no entailment label in {config.label2id}")


def run_model(name, rows, hypotheses, device_pref="auto"):
    """Load one checkpoint, score every row under every hypothesis wording."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if device_pref == "auto" else device_pref

    print("\n" + "=" * 72)
    print(f"CANDIDATE — {name}   [{device}]")
    print("=" * 72)

    before = 0
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            name, dtype=dtype).to(device).eval()
    except Exception as exc:                      # OOM, network, missing files
        print(f"  SKIPPED — could not load: {type(exc).__name__}: {exc}")
        return None
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    entail = entailment_index(model.config)
    print(f"  loaded in {load_s:.1f}s   {params/1e6:.0f}M params   "
          f"entailment slot {entail} of {model.config.label2id}")

    premises = [render(row) for row in rows]
    results = {}

    for key, (yes_hyp, no_hyp) in hypotheses.items():
        probs, latencies = [], []
        try:
            for premise in premises:
                t0 = time.perf_counter()
                # Both hypotheses go through as one batch of two, so the
                # comparison costs a single forward pass.
                batch = tok([premise, premise], [yes_hyp, no_hyp],
                            return_tensors="pt", truncation=True,
                            max_length=256, padding=True).to(device)
                with torch.no_grad():
                    logits = model(**batch).logits
                pair = logits[:, entail].float()         # entailment logit of each
                p_yes = float(torch.softmax(pair, dim=0)[0])
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
                probs.append(p_yes)
        except Exception as exc:
            print(f"  hypothesis {key!r} failed: {type(exc).__name__}: {exc}")
            continue

        # Sweep the threshold rather than guessing one. Reported best-F1, but
        # also the best threshold that keeps precision at or above 95%, because
        # answering ambient speech is the expensive mistake.
        best, best_safe = None, None
        for i in range(1, 100):
            thr = i / 100
            preds = [p >= thr for p in probs]
            m = metrics(rows, preds)
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m)
            if m["precision"] >= 0.95 and (best_safe is None
                                           or m["recall"] > best_safe[1]["recall"]):
                best_safe = (thr, m)

        mean_ms = statistics.mean(latencies)
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95) - 1]
        results[key] = {"probs": probs, "best": best, "best_safe": best_safe,
                        "mean_ms": mean_ms, "p95_ms": p95_ms}

        thr, m = best
        preds = [p >= thr for p in probs]
        report(f"hypothesis {key!r} — best F1 at threshold {thr:.2f}", rows, preds,
               f"[{mean_ms:.0f} ms mean, {p95_ms:.0f} ms p95]")
        if best_safe:
            sthr, sm = best_safe
            print(f"    at >=95% precision: threshold {sthr:.2f} keeps "
                  f"{sm['recall']:.1%} recall ({sm['fn']} real requests missed)")
        else:
            print("    never reaches 95% precision at any threshold")
        print("    by category: " + "  ".join(
            f"{tag} {ok}/{n}" for tag, (ok, n) in sorted(per_tag(rows, preds).items())))
        for kind, row in failures(rows, preds, limit=8):
            print(f"      {kind:<15} {row['id']}  {row['utterance'][:64]!r}")

    if device == "cuda":
        peak = (torch.cuda.max_memory_allocated() - before) / 1024**2
        print(f"\n  peak VRAM for this model: {peak:.0f} MiB")
        results["_vram_mib"] = peak
    results["_params"] = params
    results["_load_s"] = load_s

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# Fusion: the two approaches fail in opposite directions, so try them together
# --------------------------------------------------------------------------

# Pure acknowledgement. Aimed at JANET, but asking for nothing — the correct
# reply is silence, and `ai_core/addressing.py` already says so in its prompt.
# NLI cannot get this right, because "thanks" genuinely IS talking to the
# assistant; the hypothesis is true and the policy still says stay quiet. Five
# words of stop-list beat a 435M-parameter model on this one category.
_ACK = {"thanks", "thank you", "thanks janet", "ok", "okay", "got it", "cool",
        "nice", "great", "alright", "sure", "fine", "cheers", "perfect"}


def is_ack(text):
    return text.lower().strip().strip(".!?,") in _ACK


def run_fusion(rows, probs):
    """Compare the heuristic, the NLI model, and several ways of combining them.

    The per-category tables show the two disagreeing in a very particular way:
    the keyword scorer owns direct commands and the name, the NLI model owns
    conversation — follow-ups, answers, and two people talking to each other.
    A rule that takes each one's strength should beat both, and this prints
    whether it actually does rather than assuming it.
    """
    from intent.normalize import normalize
    from intent.scorer import score, THRESHOLD

    print("\n" + "=" * 72)
    print("FUSION — heuristic and NLI together")
    print("=" * 72)

    ling = [score(normalize(r["utterance"]))[0] for r in rows]
    acks = [is_ack(r["utterance"]) for r in rows]

    def sweep(make_prediction, label):
        best = None
        for i in range(1, 100):
            thr = i / 100
            preds = [make_prediction(thr, n, h, a)
                     for n, h, a in zip(probs, ling, acks)]
            m = metrics(rows, preds)
            if best is None or m["f1"] > best[1]["f1"]:
                best = (thr, m, preds)
        thr, m, preds = best
        print(f"\n  {label}  (threshold {thr:.2f})")
        print(f"    accuracy {m['accuracy']:.1%}   precision {m['precision']:.1%}   "
              f"recall {m['recall']:.1%}   F1 {m['f1']:.3f}   "
              f"FP {m['fp']}  FN {m['fn']}")
        print("    " + "  ".join(f"{t} {ok}/{n}"
                                 for t, (ok, n) in sorted(per_tag(rows, preds).items())))
        return m["f1"], label, thr

    out = []
    out.append(sweep(lambda t, n, h, a: n >= t, "NLI alone"))
    out.append(sweep(lambda t, n, h, a: n >= t and not a,
                     "NLI, with acknowledgements vetoed"))
    # OR — either signal is enough. Recovers the commands NLI drops, at the
    # cost of every ambient sentence the keyword list mistakes for a request.
    out.append(sweep(lambda t, n, h, a: (n >= t or h >= THRESHOLD) and not a,
                     "NLI OR heuristic, acks vetoed"))
    # AND — both must agree. Precision-first; this is the shape to pick if
    # answering the television is the thing you most want to stop.
    out.append(sweep(lambda t, n, h, a: n >= t and h >= THRESHOLD and not a,
                     "NLI AND heuristic, acks vetoed"))
    # The name is the one signal NLI cannot learn and the heuristic gets for
    # free: nobody says "Janet" to another person in this house.
    out.append(sweep(lambda t, n, h, a: (n >= t or h >= 90) and not a,
                     "NLI OR strong heuristic (name+intent), acks vetoed"))
    print("\n  best fusion rule: " + max(out)[1] +
          f"  (F1 {max(out)[0]:.3f}, threshold {max(out)[2]:.2f})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated HF checkpoints to benchmark")
    ap.add_argument("--baseline-only", action="store_true",
                    help="just score the keyword heuristic; downloads nothing")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--hypothesis", default=None,
                    help="only test this hypothesis key")
    ap.add_argument("--live", action="store_true",
                    help="score with intent/nli_scorer.py as shipped, instead "
                         "of benchmarking raw checkpoints")
    ap.add_argument("--fusion", action="store_true",
                    help="also compare ways of combining NLI with the heuristic")
    ap.add_argument("--extra-negatives", action="store_true",
                    help="add the held-out NONE class as extra ambient speech, "
                         "for a precision estimate that isn't based on 40 rows")
    args = ap.parse_args()

    rows = load_eval()
    if args.extra_negatives:
        extra = load_extra_negatives()
        rows += extra
        print(f"+ {len(extra)} extra negatives from the held-out NONE class")
    n_yes = sum(1 for r in rows if r["addressed"])
    print(f"eval set: {len(rows)} utterances — {n_yes} addressed, "
          f"{len(rows) - n_yes} ambient")

    base_m, _ = run_baseline(rows)
    if args.baseline_only:
        return 0

    if args.live:
        run_live(rows)
        return 0

    hypotheses = HYPOTHESES
    if args.hypothesis:
        hypotheses = {args.hypothesis: HYPOTHESES[args.hypothesis]}

    summary = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        res = run_model(name, rows, hypotheses, args.device)
        if not res:
            continue
        if args.fusion:
            for key in hypotheses:
                if key in res:
                    print(f"\n### fusion using {name.split('/')[-1]} / {key}")
                    run_fusion(rows, res[key]["probs"])
        for key in hypotheses:
            if key not in res:
                continue
            thr, m = res[key]["best"]
            summary.append((name, key, thr, m, res[key]["mean_ms"],
                            res.get("_vram_mib")))

    print("\n" + "=" * 72)
    print("SUMMARY — sorted by F1, baseline first")
    print("=" * 72)
    print(f"  {'model':<44} {'hyp':<10} {'thr':>5} {'F1':>6} {'prec':>6} "
          f"{'rec':>6} {'ms':>6} {'VRAM':>7}")
    print(f"  {'intent/scorer.py (heuristic)':<44} {'-':<10} {'0.40':>5} "
          f"{base_m['f1']:>6.3f} {base_m['precision']:>6.1%} "
          f"{base_m['recall']:>6.1%} {'~0':>6} {'0':>7}")
    for name, key, thr, m, ms, vram in sorted(summary, key=lambda x: -x[3]["f1"]):
        short = name.split("/")[-1]
        print(f"  {short:<44} {key:<10} {thr:>5.2f} {m['f1']:>6.3f} "
              f"{m['precision']:>6.1%} {m['recall']:>6.1%} {ms:>6.0f} "
              f"{(f'{vram:.0f}M' if vram else '-'):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
