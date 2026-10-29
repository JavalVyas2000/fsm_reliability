# Manuscript: Engineering Applications of Artificial Intelligence (Elsevier)

## Compile
- **Overleaf:** upload the `paper/` folder (`main.tex`, `sections/`, `tables/`, `figures/`, `data/`, `refs.bib`) and compile with pdfLaTeX + BibTeX. `elsarticle` is preinstalled there.
- **Locally:** run `tectonic main.tex` in `paper/`. Tectonic is installed via scoop and fetches packages on first use.
- **Figures:** all figures are TikZ/pgfplots sources in `figures/tikz/`. The data figures read `data/*.dat`. The matplotlib PDFs in `figures/*.pdf` are no longer used.

## Structure
| File | Content |
|---|---|
| `sections/01_introduction.tex` | Motivation, central question, RQ1–RQ4, contributions; Fig. 1 (overview) |
| `sections/02_related.tex` | Related work; positioning table |
| `sections/03_problem.tex` | Routing, threshold rule, Proposition 1 (certificate), Remark 1 (retries), cost model and break-even; Fig. threshold; Algorithm 1 |
| `sections/04_signals.tex` | Signal definitions (grounding Eq.), deciding regions; Fig. grounding (schematic) |
| `sections/05_setup.tex` | Domains, CSTR figure, models, partitions figure, pre-registration and statistics |
| `sections/06_results_offline.tex` | RQ1–RQ3 and shift |
| `sections/07_closed_loop.tex` | RQ4: safety plane, R0, recovery, compute |
| `sections/08_discussion.tex` | Interpretation, deployment guide figure, threats to validity |
| `sections/09_conclusion.tex` | Conclusion |

## Rules that keep the paper coherent
1. **Every claim must appear in `claims.md`,** with its evidence and caveats. If a number changes, change the ledger first.
2. **Generated content, never edited by hand:**
   - `tables/*.tex` come from `python -m scripts.49_paper_latex`.
   - `data/*.dat` come from `python -m scripts.50_paper_figure_data`.
   - Both scripts read `paper_outputs/`. Refresh it first with `python -m scripts.44_paper_outputs` and `python -m scripts.47_closed_loop_costs --runs ...`.
3. **Schematic figures** (`overview`, `grounding`, `threshold`, `cstr`, `protocol`, `guide`) contain no measured numbers, except `grounding.tex`, whose bars are explicitly illustrative.
4. **TODO markers:** text that needs author input is marked `\todo{}` (orange).

## Open items before submission
- [ ] Authors, affiliations, CRediT roles, funding, acknowledgements.
- [ ] The complete reference for the earlier ctrl-alt-recover paper (`car_todo` in `refs.bib`).
- [ ] Verify every bibliography entry. The entries added on 2026-10-29 are marked VERIFY.
- [ ] Report the teacher-forcing agreement audit for every model and domain (`\todo` in Section 4).
- [ ] The generative-AI declaration, which Elsevier requires.
- [ ] Data availability: repository URL and a Zenodo DOI for the raw records.
- [ ] Check the EAAI abstract limit (the current abstract has 239 words), the highlights format and figure requirements in the current Guide for Authors.
- [ ] Optional: a graphical abstract. `figures/tikz/overview.tex` compiled standalone is a natural candidate.
