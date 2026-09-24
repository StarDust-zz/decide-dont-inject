# decide-dont-inject

Discrete algebra pack for the Memory Decision Layer (MDL): retrieved
memory is source; between retrieval and generation a three-signal
trust compile (relevance M, reliability R, task-risk A) decides
Active / Supp / Silent / Opt-Out. Blind RAG inject is the failure mode.

Paper: Zhang, Zhang, Zhao, Ma, Zhao — *An Interpretable Memory Decision
Controller for LLM Agents Based on Three-Signal Complementarity:
Decoupling Confidence and Consistency*, arXiv:2609.22043, 18 Sep 2026.
https://arxiv.org/abs/2609.22043

## Run

```bash
python3 experiment.py
```

Expect `7/7 assertions passed` and exit code 0.

Stdlib only (Python 3 + `math`). No numpy, no sentence-transformers,
no live LLM, no network.

## Layout

| File | Role |
|------|------|
| `mdl.py` | Orthogonal-subspace MDL geometry + `decide(M,R,A)` |
| `experiment.py` | Seven falsifier assertions |
| `DESCRIPTION.md` | Claim, falsifier, synergy, keepers |
| `LICENSE` | MIT |

## License

MIT — see `LICENSE`.

## Citation

```
@article{zhang2026mdl,
  title={An Interpretable Memory Decision Controller for LLM Agents
         Based on Three-Signal Complementarity: Decoupling Confidence
         and Consistency},
  author={Zhang, Yiming and Zhang, Jinghong and Zhao, Haoran
          and Ma, Yiren and Zhao, Chunlei},
  journal={arXiv preprint arXiv:2609.22043},
  year={2026}
}
```
