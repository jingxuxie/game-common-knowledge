# Designing Common Knowledge

This repository contains the theory, exact solvers, experiments, and manuscript for:

> **Designing Common Knowledge: Budgeted Public Observations for Decentralized Team Games**

The project studies a system designer who can make only a budgeted subset of state features public.  The selected realizations become common knowledge, after which agents act using their private observations and the public interface.

## Main results

- The optimal team value is monotone in the public feature set, but need not be submodular.
- Every finite monotone Boolean access structure can be realized as a public-observation coordination problem: authorized feature sets yield value 1 and unauthorized sets yield value 1/2.
- A shared public channel and independent private channels can have identical per-agent mutual information but sharply different team values.
- Mutual-information feature selection can obtain zero coordination gain while a lower-information feature enables perfect coordination.
- Optimal feature selection is NP-hard even for two agents with binary actions; normalized gain inherits the Max-Coverage `1-1/e` approximation threshold.
- In separable revelation games, coordination gain is monotone submodular and greedy obtains the classical `1-1/e` guarantee.
- Outside that class, ordinary marginal-value greedy can have an arbitrarily small approximation ratio.
- A joint mixed-integer program exactly optimizes both the public feature set and decentralized policies.

## Reproduce

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
pytest -q
python experiments/run_all.py
python scripts/validate_results.py
```

The experiment command regenerates all CSV/JSON outputs, publication figures, and `paper/generated_results.tex`.  No GPU or learned model is required.

To build the locally compilable preprint and supplement:

```bash
make paper
```

To build the anonymous AAAI-27 source, place the **official, unmodified** `aaai2027.sty` and `aaai2027.bst` from the AAAI-27 Author Kit in `paper/`, then run:

```bash
make aaai
```

The repository deliberately does not substitute a third-party style file for the official author kit.

## Empirical snapshot

The checked-in results were generated from fixed seeds.

| Suite | Result |
|---|---:|
| Public vs. private channels, `epsilon=0.1`, `n=10` | common-knowledge premium `0.400` |
| Information-decoy instance | MI gain ratio `0.000` |
| Greedy trap, `delta=0.01` | gain ratio `0.0101` |
| 250 separable-revelation instances | greedy mean / minimum `0.997 / 0.945` of optimal gain |
| 40 operational emergency-response instances | value greedy `1.000`, MI `0.001`, private-conditioned MI `0.000` |
| Coverage scaling at 18 features | 155,382 subsets; joint MILP about `100x` faster in this run |

Timing results are illustrative and machine dependent; all value comparisons are exact up to solver tolerance.

## Repository map

- `src/common_knowledge/`: finite-game model, information objectives, exact MILP, benchmark generators, and the separable-revelation subclass.
- `tests/`: exact theorem and solver regression tests.
- `experiments/run_all.py`: all reported experiments and figures.
- `paper/main.tex`: anonymous AAAI-27 entry point (requires the official author kit).
- `paper/preprint.tex`: locally compilable two-column preprint.
- `paper/supplement.tex`: complete proof supplement.
- `scripts/validate_results.py`: machine-checks all deterministic numerical claims and generated manuscript macros.
- `notes/proofs.md`: theorem ledger and proof map.
- `notes/novelty-audit.md`: positioning against the closest literatures.
- `notes/submission-checklist.md`: final AAAI submission checks.

## Scope and limitations

The paper focuses on finite, one-shot, common-payoff team games and a fixed menu of public features.  It does not claim that mutual information is generally a bad sensing objective; rather, it proves that information quantity alone does not characterize decentralized coordination value.  Dynamic public-interface design, privacy constraints, strategic reporting, and model uncertainty are left for future work.
