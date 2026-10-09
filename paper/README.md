# Manuscript: Engineering Applications of Artificial Intelligence (Elsevier)

## Compile
- **Overleaf:** upload the `paper/` folder (`main.tex`, `sections/`, `tables/`, `figures/`, `refs.bib`), then compile with pdfLaTeX + BibTeX. `elsarticle` is preinstalled there.
- **Locally:** needs a TeX distribution (MiKTeX or TeX Live), which this laptop does not have.

## Rules that keep the paper coherent
1. **Every claim must appear in `claims.md`,** with its evidence and caveats. If a number changes, change the ledger first.
2. **Tables in `tables/` and figures in `figures/` are generated:**
   - Run `python -m scripts.49_paper_latex` after any result changes.
   - That script depends on `paper_outputs/`. Refresh it first with `python -m scripts.44_paper_outputs` and `python -m scripts.47_closed_loop_costs --runs ...`.
   - Never edit a generated table by hand.
3. **Pending numbers:** values that depend on the Qwen2.5-7B first-proposal-only closed-loop run are wrapped in `\pending{}` (shown in red).
4. **TODO markers:** text that needs author input is marked `\todo{}` (orange).

## Open items before submission
- [ ] Fill the `\pending{}` values from the Qwen2.5-7B R0 run; regenerate the tables.
- [ ] Authors, affiliations, CRediT roles, funding, acknowledgements.
- [ ] The complete reference for the earlier ctrl-alt-recover paper (`car_todo` in `refs.bib`).
- [ ] Verify every bibliography entry; add 2–3 references on LLMs for process control.
- [ ] The generative-AI declaration, which Elsevier requires.
- [ ] Data availability: repository URL and a Zenodo DOI for the raw records.
- [ ] Check the EAAI word and figure limits and the abstract length in the current Guide for Authors.
- [ ] Optional: a graphical abstract (Elsevier recommends one).
