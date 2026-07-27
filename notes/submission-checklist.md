# AAAI-27 submission checklist

## Scientific

- [x] Formal finite model and normalized coordination gain.
- [x] Deterministic sufficiency and monotonicity.
- [x] Non-submodularity witness.
- [x] Arbitrary monotone access-structure theorem.
- [x] Equal-mutual-information public/private separation.
- [x] Mutual-information decoy.
- [x] Max-Coverage hardness and approximation threshold.
- [x] Separable-revelation submodularity theorem.
- [x] Arbitrarily bad marginal-greedy construction.
- [x] Exact feature-and-policy MILP with two-way proof.
- [x] Exact tests and reproducible experiments.
- [x] Operational emergency-response benchmark without nuisance variables.
- [ ] Independent human verification of all proofs and claim wording.
- [ ] Final similarity/plagiarism check and reference audit.

## Artifact

- [x] Fixed seeds and generated CSV/JSON outputs.
- [x] Unit tests compare joint MILP with exhaustive optimization.
- [x] CPU-only requirements.
- [x] GitHub Actions test, reproduction, and numerical-claim validation workflow.
- [x] README and license.
- [x] Anonymous code package contains no author names in content.
- [ ] Remove Git history/account metadata when creating the anonymous code ZIP.

## AAAI formatting

- [x] Anonymous `paper/main.tex` entry point.
- [x] Seven-page-focused main manuscript plus separate supplement.
- [x] External PDF figures; no TikZ/pgfplots dependency.
- [x] No `hyperref`, geometry, balance, or spacing hacks in AAAI source.
- [x] AI-assisted development disclosure included in the manuscript.
- [ ] Replace/check `aaai2027.sty` and `aaai2027.bst` against the latest official Author Kit immediately before submission.
- [ ] Compile with PDFLaTeX and inspect the official-format page count.
- [x] Run `pdffonts` on the checked-in preprint and supplement; all fonts embedded and no Type 3 fonts.
- [ ] Run PDF metadata/anonymity check.
- [ ] Upload the official reproducibility checklist separately.
- [ ] Re-download all OpenReview files and verify exact versions.

## Metadata to fill manually

- [ ] Author list and ordering in OpenReview.
- [ ] Final abstract and TL;DR in OpenReview.
- [ ] Primary/secondary subject areas.
- [ ] Reviewer nomination requirements.
- [ ] Conflict-of-interest domains.
