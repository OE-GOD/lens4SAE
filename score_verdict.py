#!/usr/bin/env python3
"""Standalone pre-registered Phase B/C verdict scorer.

Reads frozen Phase A metadata plus Phase B/C JSON results and prints the full
pre-registered verdict. This file intentionally does no model work.
"""

import json
import os
from math import isfinite

from phase_bc import normalize_loaded_results, read_depth_from


LAYERS = [0, 4, 8, 12, 16, 20, 24]
VOTE_ARMS = ["comparison", "capitals", "parity"]
ANCHOR_ARM = "arith"
CONTROL_ARM = "sentiment"
ALL_ARMS = VOTE_ARMS + [ANCHOR_ARM, CONTROL_ARM]

PREDICTIONS = {
    "comparison": ("MID [L8-L16]", "EARLY-MID [L4-L12]", "YES (+ backfire at read-committed layers)"),
    "capitals": ("MID [L8-L16]", "at-or-below read", "most likely to BREAK reversal"),
    "parity": ("EARLY-MID [L4-L12]; if < L8 unevaluable", "EARLY", "only if evaluable"),
    "arith": ("[L20-L24]", "[L12-L18]", "must reproduce"),
    "sentiment": ("L0", "[L15-L18]", "must NOT reverse"),
}


def fmt_layer(layer):
    return "none" if layer is None else f"L{layer}"


def fmt_bool(value):
    return "TRUE" if value else "FALSE"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def finite_number(value):
    return isinstance(value, (int, float)) and isfinite(value)


def peak_layer_and_value(values):
    present = [(L, values[L]) for L in LAYERS if L in values and finite_number(values[L])]
    if not present:
        return None, None
    return max(present, key=lambda item: (item[1], -item[0]))


def first_crossing_depth(values):
    peak_layer, peak = peak_layer_and_value(values)
    if peak is None or peak < 0.80:
        return None, peak
    if all(L in values for L in LAYERS):
        depth, _ = read_depth_from(values, LAYERS)
        return depth, peak
    thr = 0.5 + 0.95 * (peak - 0.5)
    for L in LAYERS:
        if L in values and values[L] >= thr:
            return L, peak
    return None, peak


def full_read_depth(read_result):
    sae = read_result.get("sae", {})
    depth, peak = first_crossing_depth(sae)
    peak_layer, _ = peak_layer_and_value(sae)
    reasons = []

    if peak is None:
        reasons.append("missing SAE AUROC values")
    elif peak < 0.80:
        reasons.append(f"peak SAE AUROC {peak:.3f} < 0.80")

    perm_p = read_result.get("perm_p", {}).get(peak_layer) if peak_layer is not None else None
    if not finite_number(perm_p) or perm_p >= 0.05:
        reasons.append(f"peak-layer perm_p {perm_p!r} is not < 0.05")

    ci = read_result.get("boot_ci", {}).get(peak_layer) if peak_layer is not None else None
    ci_lower = ci[0] if isinstance(ci, (list, tuple)) and ci else None
    if not finite_number(ci_lower) or ci_lower <= 0.5:
        reasons.append(f"peak-layer boot_ci lower {ci_lower!r} is not > 0.5")

    embed = read_result.get("embed_auroc")
    if not finite_number(embed) or embed > 0.6:
        reasons.append(f"embed_auroc {embed!r} is not <= 0.6")

    l0 = sae.get(0)
    leaky = finite_number(l0) and l0 > 0.65
    if not finite_number(l0):
        reasons.append("L0 SAE AUROC missing")
    elif leaky:
        reasons.append(f"LEAKY: L0 SAE AUROC {l0:.3f} > 0.65")

    defined = depth is not None and not reasons
    return {
        "defined": defined,
        "depth": depth if defined else None,
        "threshold_depth": depth,
        "peak": peak,
        "peak_layer": peak_layer,
        "perm_p": perm_p,
        "boot_ci": ci,
        "embed_auroc": embed,
        "l0": l0,
        "leaky": leaky,
        "reasons": reasons,
    }


def raw_read_depth(read_result):
    depth, peak = first_crossing_depth(read_result.get("raw", {}))
    return {"defined": depth is not None, "depth": depth, "peak": peak}


def steer_pass(entry):
    dose = entry.get("dose") or []
    nulls = entry.get("nulls") or []
    last = dose[-1] if dose else None
    z = entry.get("z")
    sustained = bool(entry.get("sustained"))
    return (
        finite_number(last)
        and nulls
        and all(finite_number(n) for n in nulls)
        and last > max(nulls)
        and finite_number(z)
        and z >= 3
        and sustained
        and last > 0
    )


def steer_summary(steer_result):
    missing = [L for L in LAYERS if L not in steer_result]
    passing = [L for L in LAYERS if L in steer_result and steer_pass(steer_result[L])]
    return {
        "missing": missing,
        "passing": passing,
        "depth": passing[0] if passing else None,
    }


def stage0_gate(phase_a, phase_a2, arm):
    key = {"arith": "arith_anchor", "sentiment": "sentiment_control"}.get(arm, arm)
    source = phase_a.get(key, {})
    if arm == "arith" and not source:
        return phase_a2.get("anchor_choice") == "fixed16" and "arith_fixed16" in phase_a2, "phase_a2 arith_fixed16"
    return bool(source.get("gate")), f"phase_a_results.{key}.gate"


def is_non_reversed(read_depth, steer_depth, read_committed_passes):
    if read_depth is None or steer_depth is None:
        return False
    return steer_depth >= read_depth or bool(read_committed_passes)


def read_committed_passing_layers(read_depth, passing_layers):
    if read_depth is None:
        return []
    return [L for L in passing_layers if L >= read_depth]


def backfire_candidates(read_depth, steer_result):
    if read_depth is None:
        return []
    hits = []
    for L in LAYERS:
        entry = steer_result.get(L)
        if not entry or L < read_depth:
            continue
        ci4 = entry.get("ci4_mean")
        value = ci4[1] if isinstance(ci4, (list, tuple)) and len(ci4) > 1 else None
        if finite_number(value) and value < 0:
            hits.append((L, value))
    return hits


def bin_for_depth(depth):
    if depth is None:
        return "undefined"
    if depth == 0:
        return "L0"
    if 8 <= depth <= 16:
        return "MID [L8-L16]"
    if 4 <= depth <= 12:
        return "EARLY-MID [L4-L12]"
    if 12 <= depth <= 18:
        return "[L12-L18]"
    if 20 <= depth <= 24:
        return "[L20-L24]"
    return f"L{depth}"


def print_controls(controls):
    print("\n=== CONTROLS ===")
    unigram = controls.get("unigram_L0_auroc")
    print(f"unigram_L0_auroc: {unigram!r} (expected approximately 1.0)")
    shuffled = sorted((k, v) for k, v in controls.items() if k.startswith("shuffled_"))
    if not shuffled:
        print("shuffled_*: missing")
    for key, value in shuffled:
        print(f"{key}: {value!r} (expected approximately 0.5; single noisy draw)")


def main():
    missing_files = [p for p in ("phase_bc_results.json", "phase_a_results.json", "phase_a2_results.json") if not os.path.exists(p)]
    if missing_files:
        print("Missing required result files: " + ", ".join(missing_files))
        return

    phase_a = load_json("phase_a_results.json")
    phase_a2 = load_json("phase_a2_results.json")
    results = normalize_loaded_results(load_json("phase_bc_results.json"))

    print("=== PRE-REGISTERED VERDICT SCORER ===")
    print("Clause 3(c) note: matched raw-resid ordering is operationalized as rd_raw using the same peak>=0.80")
    print("and first-crossing rule; if raw peak < 0.80 or rd_raw is undefined, the raw clause fails conservatively.")
    print("Capitals caveat: exact-enumeration permutation p has a structural floor around 2/32 plus +1")
    print("smoothing, so it may be unable to clear 0.05; this is a known pre-registered-power limitation.")

    rows = {}
    incomplete = {}
    for arm in ALL_ARMS:
        if arm not in results.get("read", {}) or arm not in results.get("steer", {}):
            incomplete[arm] = LAYERS
            continue

        read_info = full_read_depth(results["read"][arm])
        raw_info = raw_read_depth(results["read"][arm])
        steer_info = steer_summary(results["steer"][arm])
        committed = read_committed_passing_layers(read_info["depth"], steer_info["passing"])

        clause_a = steer_info["depth"] is not None and read_info["depth"] is not None and steer_info["depth"] <= read_info["depth"] - 4
        clause_b = read_info["depth"] is not None and not committed
        clause_c = raw_info["defined"] and steer_info["depth"] is not None and steer_info["depth"] <= raw_info["depth"] - 4
        stage_gate, stage_source = stage0_gate(phase_a, phase_a2, arm)
        evaluability_failures = []
        if not stage_gate:
            evaluability_failures.append(f"stage-0 gate not passed ({stage_source})")
        if not read_info["defined"]:
            evaluability_failures.append("read-depth undefined: " + "; ".join(read_info["reasons"]))
        elif read_info["depth"] not in (8, 12, 16, 20):
            evaluability_failures.append(f"read-depth {fmt_layer(read_info['depth'])} not in [L8, L20]")
        if steer_info["depth"] is None:
            evaluability_failures.append("steer-depth undefined")

        rows[arm] = {
            "read": read_info,
            "raw": raw_info,
            "steer": steer_info,
            "committed": committed,
            "clauses": (clause_a, clause_b, clause_c),
            "stage_gate": stage_gate,
            "stage_source": stage_source,
            "evaluable": not evaluability_failures,
            "evaluability_failures": evaluability_failures,
            "pass": clause_a and clause_b and clause_c,
            "non_reversed": is_non_reversed(read_info["depth"], steer_info["depth"], committed),
            "backfire": backfire_candidates(read_info["depth"], results["steer"][arm]),
        }
        if steer_info["missing"]:
            incomplete[arm] = steer_info["missing"]

    print("\n=== PER-ARM TABLE ===")
    header = "arm          read-depth  rd_raw    steer-depth  PASS(a) PASS(b) PASS(c) PASS"
    print(header)
    print("-" * len(header))
    for arm in ALL_ARMS:
        if arm in incomplete:
            awaiting = ", ".join(fmt_layer(L) for L in incomplete[arm])
            print(f"{arm:<12} INCOMPLETE (awaiting layers {awaiting})")
            continue
        row = rows[arm]
        read = row["read"]
        raw = row["raw"]
        clauses = row["clauses"]
        print(
            f"{arm:<12} {fmt_layer(read['depth']):<11} {fmt_layer(raw['depth']):<9} "
            f"{fmt_layer(row['steer']['depth']):<12} "
            f"{fmt_bool(clauses[0]):<7} {fmt_bool(clauses[1]):<7} {fmt_bool(clauses[2]):<7} {fmt_bool(row['pass'])}"
        )
        if not read["defined"]:
            print(f"  read-depth UNDEFINED: {'; '.join(read['reasons'])}")
        if row["committed"]:
            print(f"  read-committed passing layers: {', '.join(fmt_layer(L) for L in row['committed'])}")
        if arm in VOTE_ARMS:
            if row["evaluable"]:
                print("  evaluability: TRUE (stage-0 gates already passed in Phase A)")
            else:
                print("  evaluability: FALSE -> replacement list; " + "; ".join(row["evaluability_failures"]))

    print("\n=== INSTRUMENT PRECONDITIONS ===")
    instrument_failures = []
    sentiment = rows.get(CONTROL_ARM)
    arith = rows.get(ANCHOR_ARM)
    if sentiment:
        s_read = sentiment["read"]["depth"]
        s_steer = sentiment["steer"]["depth"]
        s_mid_pass = any(12 <= L <= 20 for L in sentiment["steer"]["passing"])
        s_reversal = sentiment["pass"]
        ok = s_read == 0 and s_mid_pass and not s_reversal
        print(f"sentiment: read-depth {fmt_layer(s_read)}, steer pass L12-L20 {fmt_bool(s_mid_pass)}, reversal {fmt_bool(s_reversal)} -> {fmt_bool(ok)}")
        if not ok:
            instrument_failures.append("sentiment control failed")
    if arith:
        a_read = arith["read"]["depth"]
        a_steer = arith["steer"]["depth"]
        ok = a_read in (20, 24) and a_steer in (12, 16)
        print(f"arith anchor: read-depth {fmt_layer(a_read)} in [L20,L24], steer-depth {fmt_layer(a_steer)} in [L12,L18] -> {fmt_bool(ok)}")
        if not ok:
            instrument_failures.append("arithmetic anchor failed")
    print_controls(results.get("controls", {}))

    print("\n=== BACKFIRE CANDIDATES (informational only) ===")
    for arm in ALL_ARMS:
        if arm in rows:
            hits = rows[arm]["backfire"]
            text = ", ".join(f"L{L} ci4_mean[1]={value:.4g}" for L, value in hits) if hits else "none"
            print(f"{arm}: {text}")

    print("\n=== PREDICTION BINS ===")
    for arm in ALL_ARMS:
        if arm in rows:
            observed_read = bin_for_depth(rows[arm]["read"]["depth"])
            observed_steer = bin_for_depth(rows[arm]["steer"]["depth"])
        else:
            observed_read = observed_steer = "incomplete"
        pred_read, pred_steer, pred_rev = PREDICTIONS[arm]
        print(f"{arm}: predicted read {pred_read}; observed {observed_read}. predicted steer {pred_steer}; observed {observed_steer}. prereg reversal note: {pred_rev}")

    if incomplete:
        print("\n=== AGGREGATE VERDICT ===")
        for arm, missing in incomplete.items():
            print(f"{arm}: INCOMPLETE (awaiting layers {', '.join(fmt_layer(L) for L in missing)})")
        print("Aggregate verdict refused until all arms are complete.")
        return

    if instrument_failures:
        print("\n=== AGGREGATE VERDICT ===")
        print("INSTRUMENT FAILURE: " + "; ".join(instrument_failures))
        print("Nothing is scoreable under the pre-registered instrument preconditions.")
        return

    evaluable_votes = [arm for arm in VOTE_ARMS if rows[arm]["evaluable"]]
    pass_votes = [arm for arm in evaluable_votes if rows[arm]["pass"]]
    non_reversed_votes = [arm for arm in evaluable_votes if rows[arm]["non_reversed"]]

    print("\n=== AGGREGATE VERDICT ===")
    if len(evaluable_votes) == 3 and len(pass_votes) >= 2:
        verdict = "GENERALIZES"
    elif len(non_reversed_votes) >= 2:
        verdict = "ARITHMETIC-SPECIFIC"
    else:
        verdict = "INDETERMINATE"
    print(verdict)
    print(f"evaluable votes: {', '.join(evaluable_votes) if evaluable_votes else 'none'}")
    print(f"PASS votes: {', '.join(pass_votes) if pass_votes else 'none'}")
    print(f"affirmatively non-reversed evaluable votes: {', '.join(non_reversed_votes) if non_reversed_votes else 'none'}")


if __name__ == "__main__":
    main()
