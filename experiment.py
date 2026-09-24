#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run decide-dont-inject assertions. python3 experiment.py

Discrete pack of Zhang et al., arXiv:2609.22043:
retrieved memory is source; MDL compiles M/R/A into a trust decision;
blind RAG inject is the AMMI-like lowering.
"""

from __future__ import annotations

from mdl import (
    ACTION_RANK,
    THETA0,
    THETA1,
    THETA2,
    blind_rag_inject,
    decide,
    m_only_decide,
    reliability,
)


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Shared scenario fixtures (scalars in [0,1]; R via paper reliability form)
# ---------------------------------------------------------------------------
# Safe adopt: high relevance, low conflict, low risk
SAFE = dict(M=0.92, R=reliability(0.90, 0.05), A=0.10)

# Conflict: high M, high stance-conflict φ → low R, moderate A
CONFLICT = dict(M=0.92, R=reliability(0.80, 0.85), A=0.40)

# Risk twin pair: identical M,R; A high vs low
RISK_BASE_M, RISK_BASE_R = 0.90, reliability(0.85, 0.10)
RISK_HIGH = dict(M=RISK_BASE_M, R=RISK_BASE_R, A=0.90)
RISK_LOW = dict(M=RISK_BASE_M, R=RISK_BASE_R, A=0.10)

# Blind-vs-MDL: high M would RAG-inject; conflict + high A → MDL abstains
BLIND = dict(M=0.88, R=reliability(0.75, 0.90), A=0.75)

# Relevance alone insufficient: high M, low R, moderate A
REL_ONLY = dict(M=0.95, R=reliability(0.70, 0.90), A=0.45)

# Three-signal complementarity case: conflict + high risk
COMP_FULL = dict(M=0.92, R=reliability(0.80, 0.90), A=0.85)

# Decoupling probe: M≈0 so v_wm misaligned with R-driven v_meta
# → C moderate, α low → C_final ≠ C
DECOUPLE = dict(M=0.0, R=0.9, A=0.3)


def scenario_1():
    """Blind inject ≠ MDL: high-M conflict/risk → MDL not Active; RAG would inject."""
    d = decide(**BLIND)
    rag = blind_rag_inject(BLIND["M"])
    _ok(rag is True, f"blind RAG should inject on high M={BLIND['M']}")
    _ok(
        d["action"] != "Active",
        f"MDL must not Active under conflict+risk; got {d['action']} Cf={d['C_final']:.3f}",
    )
    _ok(
        d["abstain"] or d["soft"],
        f"expected reject/abstain tier, got {d['action']}",
    )
    return {
        "blind_rag": rag,
        "mdl_action": d["action"],
        "C_final": round(d["C_final"], 4),
        "R": round(d["R"], 4),
    }


def scenario_2():
    """Risk inversion: same M,R; A_high → lower C_final and Silent/Opt-Out."""
    hi = decide(**RISK_HIGH)
    lo = decide(**RISK_LOW)
    _ok(
        hi["C_final"] < lo["C_final"],
        f"high-A Cf={hi['C_final']:.3f} should be < low-A Cf={lo['C_final']:.3f}",
    )
    _ok(
        hi["g_A"] < lo["g_A"],
        f"g_A should shrink under risk inversion: {hi['g_A']:.3f} vs {lo['g_A']:.3f}",
    )
    _ok(
        hi["action"] in ("Silent", "Opt-Out"),
        f"high-A must abstain (Silent/Opt-Out); got {hi['action']}",
    )
    _ok(RISK_HIGH["A"] >= 0.85, RISK_HIGH["A"])
    return {
        "A_high_action": hi["action"],
        "A_low_action": lo["action"],
        "Cf_high": round(hi["C_final"], 4),
        "Cf_low": round(lo["C_final"], 4),
        "gA_high": round(hi["g_A"], 4),
        "gA_low": round(lo["g_A"], 4),
    }


def scenario_3():
    """Confidence ≠ consistency: C and α independent enough that C_final ≠ C."""
    d = decide(**DECOUPLE)
    _ok(
        abs(d["C"] - d["C_final"]) > 1e-3,
        f"expected C_final ≠ C; C={d['C']:.4f} Cf={d['C_final']:.4f}",
    )
    # Explicit decoupling: either |C−α| gap or gating visibly moves the score
    _ok(
        abs(d["C"] - max(0.0, d["alpha"])) > 0.05
        or abs(d["C"] - d["C_final"]) > 0.05,
        f"decoupling too weak: C={d['C']:.3f} α={d['alpha']:.3f} Cf={d['C_final']:.3f}",
    )
    return {
        "C": round(d["C"], 4),
        "alpha": round(d["alpha"], 4),
        "C_final": round(d["C_final"], 4),
        "gap_C_Cf": round(abs(d["C"] - d["C_final"]), 4),
    }


def scenario_4():
    """Conflict reliability: high M + high φ → low R → not Active."""
    d = decide(**CONFLICT)
    _ok(CONFLICT["M"] >= 0.85, CONFLICT["M"])
    _ok(d["R"] < 0.2, f"expected low R under conflict; got R={d['R']:.3f}")
    _ok(
        d["action"] != "Active",
        f"conflict must not Active; got {d['action']} Cf={d['C_final']:.3f}",
    )
    return {
        "M": CONFLICT["M"],
        "R": round(d["R"], 4),
        "action": d["action"],
        "C_final": round(d["C_final"], 4),
    }


def scenario_5():
    """Three-signal complementarity: M-only would Active; full M+R+A abstains."""
    mono = m_only_decide(COMP_FULL["M"])
    full = decide(**COMP_FULL)
    _ok(
        mono["action"] == "Active",
        f"M-only stand-in should Active; got {mono['action']}",
    )
    _ok(
        full["action"] != "Active",
        f"full M+R+A on conflict+risk must not Active; got {full['action']}",
    )
    _ok(
        full["abstain"] or full["soft"],
        f"full gate should reject/abstain; got {full['action']}",
    )
    _ok(
        ACTION_RANK[full["action"]] < ACTION_RANK[mono["action"]],
        f"full rank {full['action']} should be < M-only {mono['action']}",
    )
    return {
        "m_only": mono["action"],
        "full": full["action"],
        "Cf_m_only": round(mono["C_final"], 4),
        "Cf_full": round(full["C_final"], 4),
    }


def scenario_6():
    """Relevance alone insufficient: high M, low R, moderate A → not Active."""
    d = decide(**REL_ONLY)
    _ok(REL_ONLY["M"] >= 0.9, REL_ONLY["M"])
    _ok(d["R"] < 0.2, d["R"])
    _ok(
        d["action"] != "Active",
        f"high-M low-R must not Active; got {d['action']}",
    )
    return {
        "M": REL_ONLY["M"],
        "R": round(d["R"], 4),
        "A": REL_ONLY["A"],
        "action": d["action"],
    }


def scenario_7():
    """Safe adopt: high M, high R, low A → Active (higher rank than conflict/risk)."""
    safe = decide(**SAFE)
    conflict = decide(**CONFLICT)
    hrisk = decide(**RISK_HIGH)
    _ok(
        safe["action"] == "Active",
        f"safe path should Active; got {safe['action']} Cf={safe['C_final']:.3f}",
    )
    _ok(safe["adopt"] is True, safe["action"])
    _ok(
        ACTION_RANK[safe["action"]] > ACTION_RANK[conflict["action"]],
        f"safe {safe['action']} rank should exceed conflict {conflict['action']}",
    )
    _ok(
        ACTION_RANK[safe["action"]] > ACTION_RANK[hrisk["action"]],
        f"safe {safe['action']} rank should exceed high-risk {hrisk['action']}",
    )
    return {
        "safe": safe["action"],
        "conflict": conflict["action"],
        "high_risk": hrisk["action"],
        "Cf_safe": round(safe["C_final"], 4),
        "thresholds": (THETA0, THETA1, THETA2),
    }


SCENARIOS = [
    ("1_blind_inject_ne_mdl", scenario_1),
    ("2_risk_inversion", scenario_2),
    ("3_confidence_ne_consistency", scenario_3),
    ("4_conflict_reliability", scenario_4),
    ("5_three_signal_complementarity", scenario_5),
    ("6_relevance_alone_insufficient", scenario_6),
    ("7_safe_adopt_path", scenario_7),
]


def main() -> int:
    passed = 0
    results = []
    for name, fn in SCENARIOS:
        try:
            detail = fn()
            passed += 1
            results.append((name, "PASS", detail))
            print(f"PASS  {name}: {detail}")
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"ERROR {name}: {e}")

    total = len(SCENARIOS)
    print(f"\n{passed}/{total} assertions passed")
    print(f"thresholds θ=({THETA0}, {THETA1}, {THETA2})  [pack-local, not paper grid]")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
