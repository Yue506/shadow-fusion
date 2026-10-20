# Research workflow

- Keep `main` and the published baseline tag stable. Do research on an exploration branch created from that baseline.
- Follow `docs/RESEARCH_PLAN.md`. Evaluation shadow masks and references must never enter `compose()` or parameter selection on the locked test set.
- Preserve mode defaults and module switches. Record any deliberate algorithm changes as new configurations.
- Run `python -m unittest discover -s tests -v`; when frozen local inputs exist, run `scripts/verify_frozen_baseline.py` before claiming packaging compatibility.
- Keep real-generation experiments separate from synthetic or reference-derived proxy candidates. Never present fixture results as model accuracy.
- Report semantic background mistakes separately from the by-construction outside-envelope pixel guarantee.
- Do not commit original datasets, model weights, credentials, absolute workstation paths, or large run archives. Use `data/` and `outputs/` for local artifacts.
- No new paid compute or credential-sharing flows are implied by a research iteration. Use existing authorized resources and the stated campaign budget.
- Checkpoint useful work before a usage limit; do not spend tokens on artificial loops just to exhaust quota.
