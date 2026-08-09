"""The addressing gate: was this said to JANET, or is it just the room?

This replaces the keyword heuristic in `scorer.py` as Layer 1, and the reason is
one measurement: whether a sentence was addressed to JANET is often **not a
property of the sentence**. Scored in isolation, "and tomorrow?" gets 0 and
"Yes." gets 0, while "did you email sarah back about the invoice" — one
housemate to another — gets 65 and was answered. No keyword list fixes that,
because the same words are right in one conversational position and wrong in
another.

So the gate now reads the CONVERSATION. An NLI (natural language inference)
model is handed the last few turns as a premise and asked which of two
hypotheses follows:

    yes -> "The last speaker is talking to the voice assistant."
    no  -> "The last speaker is talking to another person."

Both go through as a single batch of two, and the score is a softmax over just
those two entailment logits. That contrast calibrates far better than reading
one hypothesis' probability on its own.

WHAT THIS FILE KEEPS FROM THE OLD SCORER, AND WHY
-------------------------------------------------
Measured on `data/text/addressing/eval.jsonl`, the model and the heuristic fail
on **disjoint sets**. Every real command the model dropped scored high on
keywords: "turn on the kitchen lights" (NLI 0.265, keywords 65), "reply to my
last email saying I'll be there at 5" (NLI 0.025, keywords 95). NLI is weakest
on a bare imperative with no conversational context, which is the exact case
keyword matching was built for; meanwhile keywords score 0 on "and tomorrow?",
where NLI is perfect.

Deleting the heuristic on principle therefore costs real commands — fusing the
two measured F1 0.822 against pure NLI's 0.794 — so a very strong keyword score
(the name AND a real intent word) may still carry the decision alone.

Acknowledgements get a stop-list rather than a model. "thanks" genuinely IS
talking to the assistant, so the hypothesis is true while the policy still says
stay quiet; every model and wording tried scored 0/4 here. Fourteen words beat
435M parameters, and it lifted precision from 74.6% to 79.4% on its own.

Numbers, 63 addressed / 169 ambient: precision 52.6% -> 80.3%, F1 0.637 ->
0.822. Two people talking went from 6/18 to 18/18; answers to JANET's own
question from 1/8 to 7/8. Design:
`docs/superpowers/specs/2026-08-08-nli-addressing-scorer-design.md`. Re-measure
with `./venv/bin/python tools/addressing_eval.py --extra-negatives --live`.
"""
import os

from intent import scorer as keywords
from intent.normalize import normalize

# Unchanged on purpose. `dispatch.py` compares against this, adds a confidence
# bonus to it, and prints it; keeping the number means one calibration constant
# carries this whole change instead of a threshold rewrite rippling outward.
THRESHOLD = 40

# 435M params, 848 MiB of VRAM, ~10ms per utterance. It beat the base (184M),
# roberta-base (125M) and xsmall (71M) checkpoints on every hypothesis wording.
# xsmall failed in an instructive way: it read "Janet" as an ordinary person's
# name, so a direct address ENTAILED "talking to another person" and `named`
# scored 6/18. Capacity fixed that, not wording — large gets 18/18.
MODEL_NAME = os.environ.get("JANET_NLI_MODEL",
                            "MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

# The measured operating point: p >= 0.20 is "addressed". Rescaled below so it
# lands exactly on THRESHOLD, which is what lets the interface stay unchanged.
NLI_CUT = float(os.environ.get("JANET_NLI_CUT", "0.20"))

# One turn is one Person+Assistant pair. Five is what the design asked for and
# what the eval set was measured with.
MAX_TURNS = int(os.environ.get("JANET_NLI_TURNS", "5"))

# A keyword score this high in practice means the name AND a real intent word.
# Below it, the heuristic is not trusted to overrule the model.
STRONG_KEYWORD = int(os.environ.get("JANET_NLI_STRONG", "90"))

# The wording IS the question being asked — these are not cosmetic strings.
# Three pairs were measured and they differ by up to 0.07 F1 on the same model:
# an "asking the assistant to do something" framing dropped two-people-talking
# to 5/18, and one that named Janet explicitly dropped talking-about-Janet to
# 1/6. Change these only alongside a re-run of tools/addressing_eval.py.
HYPOTHESIS_YES = "The last speaker is talking to the voice assistant."
HYPOTHESIS_NO = "The last speaker is talking to another person."

# Aimed at JANET, but asking for nothing. `ai_core/addressing.py` already states
# this policy in its prompt; here it is cheap and exact.
_ACK = {"thanks", "thank you", "thanks janet", "ok", "okay", "got it", "cool",
        "nice", "great", "alright", "sure", "fine", "cheers", "perfect"}

_cache = {}  # lazily filled with {"tok", "model", "device", "entail"}


def _entailment_index(config):
    """Which output slot means 'entailment' for this checkpoint.

    MNLI-style models do NOT agree on label order — some are
    contradiction/neutral/entailment, some the reverse, and the zero-shot v2
    checkpoints ship entailment/not_entailment. Reading it from the config is
    the difference between a working gate and a perfectly inverted one, so it
    is never hard-coded.
    """
    for label, idx in config.label2id.items():
        if "entail" in label.lower():
            return int(idx)
    raise ValueError(f"no entailment label in {config.label2id}")


def _load():
    """Load (once) and return the cached tokenizer/model/device/entailment slot.

    Same singleton shape as `intent/classifier.py` and `audio/stt.py` — the
    weights load a single time, not per utterance.
    """
    if not _cache:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _cache["model"] = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, dtype=dtype).to(device).eval()
        _cache["device"] = device
        _cache["entail"] = _entailment_index(_cache["model"].config)
    return _cache


def is_ack(text):
    """A bare acknowledgement — 'thanks', 'okay', 'got it'."""
    return text.lower().strip().strip(".!?,") in _ACK


def premise(query, history=None, max_turns=MAX_TURNS):
    """Render the conversation as the script the model reads.

    Written as `Person:`/`Assistant:` lines because that is the closest thing to
    natural text these models were trained on. A bare concatenation of
    utterances gives the model no way to tell who said what, which is the entire
    question being asked.
    """
    lines = []
    if history is not None:
        # messages() is the spoken chat only, oldest first: user, assistant,
        # user, assistant... so 2 messages per turn.
        for message in history.messages()[-2 * max_turns:]:
            who = "Person" if message["role"] == "user" else "Assistant"
            lines.append(f"{who}: {message['content']}")
    lines.append(f"Person: {query}")
    return "\n".join(lines)


def _owed_an_answer(history):
    """Did JANET's own last reply ask the person something?

    This is the ONLY condition under which the conversation is shown to the
    model — see `probability` for the measurement that forced it.
    """
    if history is None:
        return False
    asked = getattr(history, "last_reply_was_question", None)
    return bool(asked()) if callable(asked) else False


def probability(query, history=None):
    """P(this was aimed at JANET), in [0, 1]. Raises if the model is unusable.

    THE CONVERSATION IS ONLY SHOWN WHEN JANET IS OWED AN ANSWER, and that is
    the most surprising line in this file, so here is the measurement.

    Put an `Assistant:` turn in the premise and the hypothesis "the last
    speaker is talking to the voice assistant" becomes nearly entailed by
    construction. The model stops judging the last line and starts noticing
    that an assistant conversation is happening. Measured, same fixture:

        utterance                                  alone   with context
        "and tomorrow?"            (real)          0.036      0.762
        "i like pizza"             (ambient)       0.125      0.731
        "can you pass me the salt" (ambient)       0.107      0.723
        "why are we going this way"(ambient)       0.065      0.672

    Everything lands in 0.60-0.76 and the classes stop being separable — on the
    labelled set, ambient speech spoken DURING an open conversation scored
    0/16. `tools/full_check.py` caught it, because it does what a real room
    does: talks over JANET seconds after it answered.

    So context is spent only where it pays. When JANET's last reply ended in a
    question we are OWED an answer, and "8 am every weekday" or "oh yeah" is
    unreadable without it — that case goes from 3/8 to 8/8. Otherwise the
    utterance is judged alone, which takes ambient-during-conversation to
    15/16.

    The follow-up case this appears to give up ("and tomorrow?" after a
    STATEMENT) is not actually lost: it falls to `dispatch._rescue`, which asks
    the full LLM and has the whole conversation. That path already existed, is
    gated on JANET having spoken within 30s, and was built for exactly this.
    A false negative there costs a second; a false positive has no safety net
    at all.
    """
    import torch

    c = _load()
    text = premise(query, history if _owed_an_answer(history) else None)
    batch = c["tok"]([text, text], [HYPOTHESIS_YES, HYPOTHESIS_NO],
                     return_tensors="pt", truncation=True, max_length=256,
                     padding=True).to(c["device"])
    with torch.no_grad():
        logits = c["model"](**batch).logits
    # One entailment logit per hypothesis; softmax over the pair is the contrast.
    pair = logits[:, c["entail"]].float()
    return float(torch.softmax(pair, dim=0)[0])


def score(query, history=None):
    """Return (total, breakdown) — the same shape `scorer.score` returned.

    `query` is the RAW transcript, not the normalized form. The old scorer
    needed lowercase and no punctuation for its word matching; an NLI model
    wants the opposite, because casing and question marks are signal.
    """
    if is_ack(query):
        return 0, [("acknowledgement", 0)]

    # The keyword score is still computed: it is the strong-signal rescue, and
    # the fallback if the model cannot run.
    ling, breakdown = keywords.score(normalize(query))

    try:
        p = probability(query, history)
    except Exception as exc:                  # missing weights, OOM, bad config
        # Degrade to the old gate rather than going deaf. Same posture as
        # `ai_core/addressing.py`: a model outage must not turn JANET off.
        print(f"⚠  NLI scorer unavailable ({type(exc).__name__}: {exc}) "
              f"— falling back to keywords")
        return ling, breakdown

    # Rescale so the measured cut-off lands exactly on THRESHOLD. p = 0.31
    # becomes 40; anything at or above the threshold is "addressed", which is
    # the comparison dispatch.py already makes.
    nli_points = min(100, round(p / NLI_CUT * THRESHOLD))

    # The keyword signals stay in the breakdown for the runtime log and for
    # dispatch's `_weakly_addressed` check, but only a very strong keyword
    # score is allowed to carry the decision on its own.
    total = max(nli_points, ling if ling >= STRONG_KEYWORD else 0)
    return total, [("nli", nli_points)] + breakdown


def is_addressed(query, history=None):
    """True if the utterance scores at or above the threshold."""
    total, _ = score(query, history)
    return total >= THRESHOLD
