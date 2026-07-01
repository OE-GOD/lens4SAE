"""Datasets + readouts for the computed-concepts generalization experiment (pre-freeze).

Items and rules come from the adversarial design review (verdict: run-with-fixes). Phase A vets these
behaviorally; after Phase A the surviving items are frozen by git commit (PREREG.md records the hash).

Design rules enforced here in code (`check_all()` aborts the run on violation):
- comparison: XOR expansion — each number pair (a,b), a>b, yields POS ["a is greater than b",
  "b is less than a"] and NEG ["b is greater than a", "a is less than b"]; within every minimal pair
  the token multiset is IDENTICAL (kills larger-first / comparator-word / position / digit-count leaks).
- capitals: region-closed derangement — every country and city appears once in EACH set; token balance
  holds at the SET level (within-pair identity is impossible for recall items). Foil + bare-pair
  association controls break the PMI==label confound.
- parity: magnitude-matched, completion-screened numbers; SET-level token balance.
- fit / probe / few-shot vocabularies disjoint per concept; no statement ends in '.' (template adds it).
"""
import random
import torch
from featurescope.concepts import ReadoutSpec

# ---------------- A: comparison-order (XOR design) ----------------
# (12,3) from the review violates its own frozen gap rule (cross-decade gap 9 < 10) -> (13,3), gap 10.
CMP_PAIRS = [(13, 3), (18, 7), (25, 12), (31, 18), (38, 25), (46, 31), (52, 38), (59, 46), (67, 52),
             (74, 59), (82, 67), (89, 74), (96, 82), (47, 42), (64, 61), (95, 91), (26, 23), (59, 53),
             (38, 32), (9, 4)]
CMP_POS = [s for a, b in CMP_PAIRS for s in (f"{a} is greater than {b}", f"{b} is less than {a}")]
CMP_NEG = [s for a, b in CMP_PAIRS for s in (f"{b} is greater than {a}", f"{a} is less than {b}")]
CMP_CLUSTER = [i // 2 for i in range(len(CMP_POS))]      # LOPO unit = number pair (4 statements)

CMP_PROBES_TRUE = ["34 is greater than 21", "87 is greater than 54", "66 is greater than 63",
                   "72 is greater than 39", "58 is greater than 55", "81 is greater than 44",
                   "45 is less than 78", "27 is less than 49", "90 is less than 94",
                   "16 is less than 68", "29 is less than 77", "80 is less than 85"]
CMP_PROBES_FALSE = ["21 is greater than 34", "54 is greater than 87", "63 is greater than 66",
                    "39 is greater than 72", "55 is greater than 58", "44 is greater than 81",
                    "78 is less than 45", "49 is less than 27", "94 is less than 90",
                    "68 is less than 16", "77 is less than 29", "85 is less than 80"]

_CMP_TMPL = "Statement: {text}. Verdict:"
CMP_FEWSHOT_4 = ("Statement: 130 is greater than 40. Verdict: correct\n"
                 "Statement: 20 is greater than 105. Verdict: incorrect\n"
                 "Statement: 15 is less than 110. Verdict: correct\n"
                 "Statement: 120 is less than 35. Verdict: incorrect\n")
CMP_FEWSHOT_2 = ("Statement: 130 is greater than 40. Verdict: correct\n"
                 "Statement: 20 is greater than 105. Verdict: incorrect\n")

def cmp_spec(few_shot):
    return ReadoutSpec(name="comparison", few_shot=few_shot, template=_CMP_TMPL,
                       pos_word=" correct", neg_word=" incorrect",
                       probes=CMP_PROBES_TRUE + CMP_PROBES_FALSE)

# ---------------- B: capitals (region-closed derangement) ----------------
CAP_FIT = [("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
           ("Japan", "Tokyo"), ("Russia", "Moscow"), ("England", "London"), ("Egypt", "Cairo"),
           ("China", "Beijing"), ("Canada", "Ottawa"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
           ("Netherlands", "Amsterdam"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
           ("South Korea", "Seoul")]
CAP_WRONG = {"France": "Rome", "Italy": "Madrid", "Spain": "Lisbon", "Portugal": "Paris",
             "Germany": "London", "England": "Moscow", "Russia": "Athens", "Greece": "Berlin",
             "Japan": "Beijing", "China": "Seoul", "South Korea": "Tokyo",
             "Netherlands": "Vienna", "Austria": "Budapest", "Hungary": "Amsterdam",
             "Egypt": "Ottawa", "Canada": "Cairo"}
CAP_POS = [f"The capital of {c} is {city}" for c, city in CAP_FIT]
CAP_NEG = [f"The capital of {c} is {CAP_WRONG[c]}" for c, _ in CAP_FIT]
# derangement cycles (cluster unit for LOPO / bootstrap): leaving out a cycle removes its cities entirely
CAP_CYCLES = [["France", "Italy", "Spain", "Portugal"], ["Germany", "England", "Russia", "Greece"],
              ["Japan", "China", "South Korea"], ["Netherlands", "Austria", "Hungary"],
              ["Egypt", "Canada"]]
_cyc_of = {c: k for k, cyc in enumerate(CAP_CYCLES) for c in cyc}
CAP_CLUSTER = [_cyc_of[c] for c, _ in CAP_FIT]

# famous-foil diagnostics (NOT pooled with fit; transfer test only) — POS true capital / NEG famous city
CAP_FOILS = [("Morocco", "Rabat", "Casablanca"), ("New Zealand", "Wellington", "Auckland"),
             ("Pakistan", "Islamabad", "Karachi"), ("Nigeria", "Abuja", "Lagos"),
             ("Kazakhstan", "Astana", "Almaty"), ("USA", "Washington", "New York"),
             ("Turkey", "Ankara", "Istanbul"), ("Australia", "Canberra", "Sydney")]
CAP_ASSOC_POS = [f"{c} {city}" for c, city in CAP_FIT]                     # bare-pair association controls
CAP_ASSOC_NEG = [f"{c} {CAP_WRONG[c]}" for c, _ in CAP_FIT]

CAP_PROBES_TRUE = [("Norway", "Oslo"), ("Ireland", "Dublin"), ("Sweden", "Stockholm"),
                   ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Czechia", "Prague"),
                   ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Cuba", "Havana"),
                   ("Iraq", "Baghdad"), ("Chile", "Santiago"), ("Iran", "Tehran")]
CAP_PROBES_FALSE = [("Norway", "Copenhagen"), ("Denmark", "Oslo"), ("Ireland", "Helsinki"),
                    ("Finland", "Dublin"), ("Sweden", "Prague"), ("Czechia", "Stockholm"),
                    ("Peru", "Santiago"), ("Chile", "Lima"), ("Iraq", "Tehran"),
                    ("Iran", "Baghdad"), ("Kenya", "Havana"), ("Cuba", "Nairobi")]
CAP_PROBE_FALLBACK = ("Belgium", "Brussels")

def cap_probe_texts(true_pairs=CAP_PROBES_TRUE, false_pairs=CAP_PROBES_FALSE):
    return ([f"The capital of {c} is {x}" for c, x in true_pairs]
            + [f"The capital of {c} is {x}" for c, x in false_pairs])

def cap_spec(probe_texts=None):
    return ReadoutSpec(
        name="capitals",
        few_shot=("Statement: The capital of Argentina is Buenos Aires. Verdict: correct\n"
                  "Statement: The capital of Argentina is Kabul. Verdict: incorrect\n"),
        template="Statement: {text}. Verdict:", pos_word=" correct", neg_word=" incorrect",
        probes=probe_texts if probe_texts is not None else cap_probe_texts())

# ---------------- C: parity (magnitude-matched, completion-screened) ----------------
PAR_EVENS = [8, 12, 20, 28, 40, 60, 64, 90]
PAR_ODDS = [11, 19, 33, 35, 59, 67, 71, 91]
PAR_POS = ([f"{n} is an even number" for n in PAR_EVENS] + [f"{n} is an odd number" for n in PAR_ODDS])
PAR_NEG = ([f"{n} is an odd number" for n in PAR_EVENS] + [f"{n} is an even number" for n in PAR_ODDS])
PAR_CLUSTER = list(range(len(PAR_POS)))                   # LOPO unit = the number (its POS+NEG pair)

PAR_PROBES = ["10 is an even number", "13 is an even number", "17 is an odd number",
              "24 is an odd number", "31 is an odd number", "16 is an even number",
              "25 is an even number", "38 is an odd number", "42 is an even number",
              "29 is an odd number", "50 is an odd number", "21 is an even number",
              "46 is an even number", "62 is an even number", "84 is an even number",
              "53 is an odd number", "75 is an odd number", "97 is an odd number",
              "44 is an odd number", "68 is an odd number", "86 is an odd number",
              "57 is an even number", "73 is an even number", "99 is an even number"]
PAR_PROBE_TRUTH = [True, False, True, False, True, True, False, False, True, True, False, False,
                   True, True, True, True, True, True, False, False, False, False, False, False]

def par_spec(probes=None):
    return ReadoutSpec(
        name="parity",
        few_shot=("Statement: 4 is an even number. Verdict: correct\n"
                  "Statement: 5 is an even number. Verdict: incorrect\n"),
        template="Statement: {text}. Verdict:", pos_word=" correct", neg_word=" incorrect",
        probes=probes if probes is not None else PAR_PROBES)

# ---------------- anchor + control ----------------
def arith_sets(n=40, seed=0):
    rng = random.Random(seed)
    pos, neg = [], []
    for _ in range(n):
        a, b = rng.randint(2, 19), rng.randint(2, 19)
        delta = rng.choice([-2, -1, 1, 2])
        pos.append(f"{a} + {b} = {a + b}"); neg.append(f"{a} + {b} = {a + b + delta}")
    return pos, neg

def arith_spec():
    from arith_steer import ARITH
    return ARITH

def sentiment_sets_and_spec():
    from featurescope import data
    from featurescope.concepts import SENTIMENT
    pos, neg = data.examples_for("sentiment")
    return pos, neg, SENTIMENT

# ---------------- shared readout ----------------
def make_readout(model, spec):
    pos_id = model.to_tokens(spec.pos_word, prepend_bos=False)[0, 0]
    neg_id = model.to_tokens(spec.neg_word, prepend_bos=False)[0, 0]
    def readout(text, fwd_hooks=None):
        prompt = spec.few_shot + spec.template.format(text=text)
        with torch.no_grad():
            lg = model.run_with_hooks(model.to_tokens(prompt), fwd_hooks=fwd_hooks or [])
        return float(lg[0, -1, pos_id] - lg[0, -1, neg_id])
    return readout

# ---------------- in-code dataset assertions ----------------
def _tok_multiset(model, text):
    return tuple(sorted(model.to_tokens(text, prepend_bos=False)[0].tolist()))

def check_all(model):
    fails = []
    for s in CMP_POS + CMP_NEG + CAP_POS + CAP_NEG + PAR_POS + PAR_NEG:
        if s.endswith("."):
            fails.append(f"statement ends in '.': {s!r}")
    # comparison: WITHIN-pair token multiset identity (the strong guarantee)
    for p, n in zip(CMP_POS, CMP_NEG):
        if _tok_multiset(model, p) != _tok_multiset(model, n):
            fails.append(f"comparison pair token mismatch: {p!r} vs {n!r}")
    # capitals + parity: SET-level token balance (within-pair identity impossible by design)
    for name, pos, neg in [("capitals", CAP_POS, CAP_NEG), ("parity", PAR_POS, PAR_NEG)]:
        mp = sorted(t for s in pos for t in _tok_multiset(model, s))
        mn = sorted(t for s in neg for t in _tok_multiset(model, s))
        if mp != mn:
            fails.append(f"{name}: set-level token multisets differ (POS vs NEG)")
    # vocabulary disjointness (fit vs probes vs few-shot), per concept
    cmp_fit_nums = {x for pair in CMP_PAIRS for x in pair}
    cmp_probe_nums = {int(w) for s in CMP_PROBES_TRUE + CMP_PROBES_FALSE for w in s.split() if w.isdigit()}
    cmp_few_nums = {130, 40, 20, 105, 15, 110, 120, 35}
    if cmp_fit_nums & cmp_probe_nums or cmp_fit_nums & cmp_few_nums or cmp_probe_nums & cmp_few_nums:
        fails.append("comparison: fit/probe/few-shot numbers overlap")
    fit_geo = {c for c, _ in CAP_FIT} | set(CAP_WRONG.values()) | {city for _, city in CAP_FIT}
    probe_geo = {c for c, _ in CAP_PROBES_TRUE + CAP_PROBES_FALSE} | {x for _, x in CAP_PROBES_TRUE + CAP_PROBES_FALSE}
    if fit_geo & probe_geo or {"Argentina", "Buenos Aires", "Kabul"} & (fit_geo | probe_geo):
        fails.append("capitals: fit/probe/few-shot geography overlaps")
    par_fit = set(PAR_EVENS + PAR_ODDS)
    par_probe = {int(s.split()[0]) for s in PAR_PROBES}
    if par_fit & par_probe or {4, 5} & (par_fit | par_probe):
        fails.append("parity: fit/probe/few-shot numbers overlap")
    return fails
