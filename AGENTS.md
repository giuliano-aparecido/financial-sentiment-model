# Project docs

Full documentation lives in dedicated files, not duplicated here:

- [`README.md`](README.md) — what this pipeline produces, the two-stage
  Task A/Task B design, Colab-only training path
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — **branch + PR is required here,
  never push directly to `main`** — see there for the exact workflow, the
  no-CI-so-run-it-yourself testing expectations, and the byte-identical
  sync requirement across the duplicated Colab/RunPod eval scripts
- [`docs/`](docs) — design/audit notes written during specific pieces of
  work (dataset fixes, training results, valuation-prompt audits, the
  two-stage redesign) rather than living reference docs; check dates and
  whether a plan there was ever executed before trusting it as current

Branch + PR, never push directly to `main` — see
[`agent-config/AGENTS.md`](agent-config/AGENTS.md) (a git submodule
shared across this repo's siblings) for the full workflow, plus the rest
of the fleet-wide conventions, loaded automatically below for Claude Code.

@agent-config/AGENTS.md
