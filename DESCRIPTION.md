# Decide, don't inject

A discrete pack of the **Memory Decision Layer (MDL)**: retrieved
memory is a source candidate, not an executable. Between retrieval and
generation sits a trust compile over relevance M, reliability R, and
task-risk A — with confidence–consistency decoupling, risk inversion,
and explicit abstention. Blind RAG inject is the AMMI-like lowering.

Source: Zhang, Zhang, Zhao, Ma, Zhao, *An Interpretable Memory Decision
Controller for LLM Agents Based on Three-Signal Complementarity:
Decoupling Confidence and Consistency*, submitted 18 Sep 2026,
[arXiv:2609.22043](https://arxiv.org/abs/2609.22043).

```
python3 experiment.py
```

No GPU. No live model. No sentence-transformers. No product import.
Pure Python + math. This is the algebra, not the paper's TruthfulQA /
HaluEval tables or calibrated θ grid search.

## The object

```
M          = relevance           (max-cosine stand-in; scalar in pack)
R          = reliability         clip(s̄_rel · (1-φ)², 0, 1); 0.5 if <2 relevant
A          = task risk           sinv_A = 1 − A   (risk inversion)
v(s)       = s·1_D + Σ_i exp(-(s−c_i)²/(2σ²)) e_i     D=16, σ=0.20
Π_k        = orthogonal projectors via QR (dims 5,5,6)
W_k        = s_k Π_k + ε_c Σ_{j≠k} Π_j               ε_c=0.10
g_A        = 0.5 + 0.5 ⟨v_a⟩                          from sinv_A encoding
v_meta     = tanh(g_A W_wm v_wm + W_r v_r + W_a v_a)
C          = clip((‖v_meta‖ − n_min)/(n_max − n_min), 0, 1)   # confidence (norm)
α          = cos(v_wm, v_meta)                                 # consistency
C_final    = C · (0.3 + 0.7 α)
action     ∈ {Active, Supp, Silent, Opt-Out} via θ=(θ0,θ1,θ2)
```

Pack-local thresholds (object is the algebra, not the paper grid):
θ = (0.85, 0.74, 0.50). Active = adopt/inject; Supp/Silent/Opt-Out =
reject or abstain (Supp = soft tier; Silent/Opt-Out = full abstain).

**Claim.** Retrieved memory is source. MDL compiles M, R, and A into a
trust decision with C–α decoupling, risk inversion, and explicit
abstention. Blind RAG injects on relevance alone — that is the
AMMI-like lowering.

**Falsifier.** If high relevance alone justified Active inject under
conflict or high risk, or if confidence C (norm) equaled consistency α
(cosine), or if risk inversion did not force abstention at high A, the
object collapses.

## What the paper claimed that this pack tests

1. Blind inject ≠ MDL: same high-M case with conflict/high A → RAG
   injects; MDL does not Active.
2. Risk inversion: identical M,R with A_high vs A_low → high A lowers
   C_final / g_A and yields Silent/Opt-Out.
3. Confidence ≠ consistency: a case where C and α diverge so
   C_final ≠ C — decoupling is real.
4. Conflict reliability: high M + high φ → low R → not Active.
5. Three-signal complementarity: M-only (R=0.5, A=0) would Active;
   full M+R+A on conflict+risk abstains.
6. Relevance alone insufficient: high M, low R, moderate A → not Active.
7. Safe adopt path: high M, high R, low A → Active, higher rank than
   conflict and high-risk cases.

## Synergy

Pairs with [compile-dont-inject](https://github.com/StarDust-zz/compile-dont-inject):
memory is source; Brief State / gate compiles the executable; AMMI
injects without a trust compile. Also
[refuse-to-certify](https://github.com/StarDust-zz/refuse-to-certify)
(abstention) and
[access-dont-dump](https://github.com/StarDust-zz/access-dont-dump)
(don't dump context). Here the compile is the MDL trust decision
between retrieval and generation.

## Keepers

If 7/7 pass: **retrieved memory is source; MDL compiles relevance,
reliability, and task-risk into a trust decision with
confidence–consistency decoupling, risk inversion, and abstention.
Blind RAG inject is the AMMI-like lowering.**
