# Repository guide

English | [简体中文](repository-guide.zh-CN.md)

This repository presents accepted research implementations, core data, and
reproduction materials. Exploratory experiments, tuning, complete raw archives,
and operational logs are maintained in the separate private research workspace.

Start with [probabilistic computing and p-bits](probabilistic-computing.md)
for concepts, principles, and history. The [benchmark overview](benchmark-overview.md)
summarizes accepted scales, timings, parallel states, and memory conventions.

## Project boundaries

Use one repository and one locked uv environment. Group configurations, scripts,
tests, data, baseline locks, and provenance by problem. Keep each Python problem
package independent; introduce shared code only when multiple projects actually
need it. Shared modules must not import problem-specific research logic.

| Path | Contents |
|---|---|
| `src/<problem_package>/` | Independent solver packages; factorization uses `sparse_factorization` |
| `configs/<problem>/` | Frozen public inputs and protocols |
| `scripts/<problem>/`, `tests/<problem>/` | Reproduction helpers and local-only software checks |
| `data/<problem>/` | Accepted core data, certificates, and versioned historical evidence |
| `baselines/<problem>/`, `provenance/<problem>/` | Baseline versions and source provenance |
| `docs/<problem>/` | Reports, methods, reproduction, and source guides |

Agent instructions (`AGENTS.md`) and software checks (`tests/`)
are local-only: Git ignores them while existing local files are retained.

The repository contains factorization, cubic spin glass, TSP, MWIS, weighted
Max-SAT, sparse/dense Max-Cut, LABS, strict bisection, and LDPC likelihood search.
The seven additional topics use independent `combinatorial_solvers/<problem>/`
subpackages with shared model/provenance utilities. Their frozen call
configurations and certificates are under `data/<panel>/rtx5090/`; see the
[reproduction guide](reproduction.md).

Changes to one project preserve other interfaces, evidence, and environments.
New applications require actual accepted results; an empty directory is not a result.

## Documentation and references

English is the primary text: `README.md` and `name.md`. Chinese translations
use `README.zh-CN.md` and `name.zh-CN.md`. Each pair has a language link at the
top. Update both together, preserving numbers, formulas, commands, evidence
boundaries, and citation identifiers. Link Chinese readers to Chinese companion
pages where available. Publication titles remain in their original language.

Only Markdown content is bilingual. Repository names, paths, directories, and
filenames (including Markdown names) use English, as do code identifiers, comments,
docstrings, command/help/log/error messages, configuration and data keys/text values,
and figure labels. Store generated-report translations in paired Markdown templates;
source and test files remain English. This rule applies to the public repository.

Documentation uses Markdown only. Figures may be SVG or PNG embedded in Markdown;
do not create PDF versions of documents or figures. Each experiment report ends
with a linked bibliography covering every source actually used in that project's
research, including background and excluded directions. Keep paper PDFs,
supplements, retrieval records, and detailed citation audits in the private
workspace. A bibliography entry does not imply adoption or replication.

The root README stays brief. Explain implementation details in project source
guides. Project GPU results use measured RTX 5090 timings, with CPU controls and
their hardware described separately.

## Maintenance and reproduction

Start maintenance by reading both repositories' applicable instructions and
checking branches, staged changes, active tasks, and file ownership. Transfer an
explicit set of accepted materials, validate it independently, then record its
source in the private workspace. Batch completion in research does not
automatically import an intermediate result here.

Ordinary software checks run in this project's locked environment. Expensive
reproduction requires its own device and budget allocation. Keep one experiment
at a time on a GPU used for timing. Network transfer and CPU verification may
overlap completed chunks without materially affecting sampling.

Local commits are authorized only while the current branch is `dev`, after
relevant checks, using English Conventional Commits and task-scoped staging.
Preserve unrelated work. Remote publication, license selection, and changes to
publication scope require the owner's applicable authorization.

## Public branch and license

The project uses the [MIT license](../LICENSE). Local `dev` retains development
history. Public `main` is a single root commit with the accepted `dev` file tree.
An authorized publication pushes that explicit branch only; no development
branches or tags are sent. Preserve the local development branch after publishing.

Accepted set-cover code is in `src/combinatorial_solvers/set_cover/`, with
117 natural calls and four replay core records in `data/set-cover/rtx5090/`.
See its [report](set-cover/report.md) and [independent reproduction entry](set-cover/source-guide.md).

The JSSP [report](jssp/report.md) retains 72 natural calls and three replays. Its [source guide](jssp/source-guide.md) provides input downloads, independent core-witness checks and isolated work replays.

Covering arrays retain 50 natural confirmation calls and three replays, plus a separate 10-call development panel and one replay. The [report](covering-array/report.md) and [source guide](covering-array/source-guide.md) describe the model, independent certificates and reproduction steps.

Coloring keeps final confirmation, development comparisons and fixed-work implementation panels separate. Original DIMACS inputs are fetched by pinned hashes. The owned TabuCol-style program is an independent CPU reference, not an upstream HEAD reproduction. See the [report](graph-coloring/experiment-report.md) and [source guide](graph-coloring/source-guide.md).

QAP retains 72 independent confirmation calls, four implementation controls and five replays, with separate Q1c development and official Ro-TS ledgers. The [report](qap/experiment-report.md) and [model/reproduction guide](qap/README.md) give complete parameters, witness checks and panel boundaries.
