# sim-viewer

A local web app for inspecting how the dsa05 methods behave in the scenario v2.0
simulation: where and why a method misses deadlines, and what one method (e.g. U)
changes relative to another (e.g. M) on the same deployment. It links the
aggregate protocol metrics (P2, P1, N*) to slot-level behaviour.

The app is a descriptive tool. It does not produce results for the paper, and it
never selects, tunes or evaluates a method on a held-out set.

Status: plan approved on 2026-10-11. Revised on 2026-10-12 with the constraints
agreed for running alongside EXP-011b (design 4f979d9, §8): the viewer slice, the
EXP-011b sets, the results gate, commit timing and the process cap. Phase 1 (results
explorer and deployment map) implemented on 2026-10-12 on branch `web/sim-viewer`;
its deviations from this plan are marked "Phase 1:" below.

## 1. Decisions

| Topic | Decision |
|---|---|
| Repository | Part of the main dsa05 repository, under `web/sim-viewer/`. Other web apps get their own subdirectories of `web/`; shared code moves to `web/shared/` only once a second app needs it. |
| Development | In a git worktree outside the main checkout (e.g. `git worktree add ../dsa05_web -b web/sim-viewer`), merged into `main` only between formal stages (§2.2). Uncommitted app files then never appear in the main checkout's `git status`. |
| Python environment | Its own uv project and `.venv` in `web/sim-viewer/`, with `dsa-period` as an editable path dependency (`../..`), numpy pinned to the root lock's version, and torch (CPU index) added in Phase 2, when policies run. The root `pyproject.toml`, `uv.lock` and `.venv` are not changed for the app. |
| Backend | FastAPI + uvicorn, bound to `127.0.0.1` only. Use `ssh -L` for remote access. |
| Frontend | Vite + React + TypeScript. Dense time series drawn on `<canvas>`. Phase 1: the explorer charts are small hand-written SVG components instead of uPlot or ECharts; they need only a scatter, a step line and interval glyphs. |
| Dependencies | Only releases at least two weeks old when locked: uv `exclude-newer = 2026-09-27`, npm `--before=2026-09-27` with exact versions. One exception: `source-map-js` 1.2.2 (2026-09-30, a build-time dependency of vite), the fix for GHSA-68fv-2mgg-jv7q. |
| Urgency, summary and yield internals | Recomputed in the app from the recorded observations (pure functions of `dsa_period.urgency`). `src/` is not changed. A read-only trace hook in `src/` is a later option and needs tests showing that replays stay identical. |
| Instances the app may simulate | The allowlist in §2.1. |

## 2. Rules

### 2.1 Held-out sets

A viewer that can simulate any instance could read a protected set by accident.
`server/guards.py` therefore enforces an explicit allowlist. Every request that runs
a policy goes through it, and the check is covered by tests.

Allowed (split `dev` only):

- dev 0-63 (phase-0 magnitude check, already used);
- dev 1000-1399 (EXP-009 step 0 and the information-boundary probes, already used);
- `exp010-eval512` (dev 4000-4511, already read);
- `exp011-eval512` (dev 5000-5511), only after the EXP-011 report is committed;
- the viewer slice, dev 5512-5999 (488 deployments, unregistered as of 2026-10-12),
  registered in `splits_v2` in Phase 2. No experiment ever uses it.

Everything else is refused. The guard additionally hard-codes these, so that a
mistake in the allowlist cannot open them:

- `test`, `v2-confirm1` (reserve 30000-30499, the final confirmation set, unread),
  `v2-confirm2` (`splits_v2.HELD_OUT`) and every other reserve index (31000+);
- `exp008-screen256` (selection only, never diagnostics) and `exp009-screen256`
  (EXP-011 checkpoint selection);
- the EXP-011b sets until the EXP-011b report is committed: `exp011b-eval512`
  (dev 6000-6511) and the whole block reserved for `exp011b-stress` (dev
  7000-9999, although the stress set reads only some deployments in it).

Results that already exist in `outputs/` read no new instance, so the app may
show them (including test or v2-confirm2 results), but only for experiments whose
report is committed. An experiment that is still running, or whose report is not
committed yet, is not shown at all: its interim results would bypass the approval
gates of the pre-registered workflow. Regenerating a deployment's geometry for the
map (`Scenario.instance`, no policy run) is allowed only for instances that
already have a formal record of a shown experiment.

### 2.2 Formal runs

- Formal entry points refuse a dirty tree (`git status --porcelain`), and untracked
  files count. Do not leave app files uncommitted in the main checkout while a
  formal run is active, or one is about to be launched. Work in the worktree
  instead.
- Merge the web branch into `main`, or commit anything to `main`, only between
  formal stages. A new HEAD during a stage makes the later evaluation chunks record
  a different commit. Results do not change (the app never touches `src/` or
  `configs/`), but the provenance becomes mixed.
- Never check out the web branch in the main checkout: it would swap files under
  running processes.
- The worktree's `.venv` imports the worktree's own copy of `src/`, so app work
  cannot change what an experiment imports. That copy can fall behind `main`:
  merge `main` into the web branch between stages, and show the commit a replay
  ran on next to its exactness check (§4.3).
- `outputs/` is read only. The app writes only to its own gitignored cache
  (`web/sim-viewer/.cache/`).
- Simulation workers: 1-2 processes by default, and at most 4 while a formal stage
  runs (formal runs use 28 of the 32 cores). Results are seeded, so running at the
  same time as an experiment only slows both down.
- App simulations keep the machine load up, which hides a stalled driver from a
  stall check based on load average (as on 2026-10-11). Waiters for formal runs
  should measure the CPU use of the driver's own process tree instead.

## 3. Views

1. **Results explorer.** Built from `summary.json`, `episodes.json` and
   `analysis.json`; runs no simulation.
   - A method x N table of P2 and P1, taken from the pre-registered analysis as
     written, never recomputed differently. Phase 1: `analysis.json` has no
     per-method intervals, so the table shows point estimates (also p90 and the
     all-UE mean, and N* on the grid). Intervals appear only for the paired
     differences.
   - The pre-registered comparisons (D_N, E_N, content, mechanism, receive
     conditions, guard and sensitivity), each row with its difference, 95%
     interval, ratio of means and whether the interval excludes 0.
   - A paired scatter of each deployment's worst-UE violation rate, method A
     against method B, on log10(w + 10^-4).
   - Distributions (ECDF) of the worst-UE rate, and the deployments with the
     largest A - B differences.
   - The P2 recomputed from the episode records is shown next to the analysis
     value; a mismatch is flagged.
   - Clicking a deployment opens views 2-5.
2. **Deployment map.** The AP, the 100 m coverage disk, and the UEs, with distance and
   shadowing on hover. Phase 1: colour is reserved for methods A and B across the
   page, so the traffic class (10/20/40-slot periods) is the marker shape, rings
   mark each method's worst UE, and a per-UE table next to the map gives the
   violation rates (seed-layer means, per seed on hover) and undeliverable packets.
3. **Replay timeline** (the core view). All panels share one time cursor:
   - **Subchannel occupancy:** K = 4 rows by slot, showing which UEs transmit, their
     power level and SINR, and collisions.
   - **Per-UE packet rows:** arrival-to-deadline windows, transmission slots,
     completions and violations, slack and urgency, yields. Packets that are
     undeliverable even alone (`capacity.undeliverable`) are marked.
   - **Per-UE gain heatmap:** gain on each subchannel over time (Jakes fading at
     10 Hz), which shows the "weak even when alone" cases.
   - **Overview strip:** violation density over the 20,200-slot horizon, for
     navigation.
4. **Paired comparison.** One deployment with the same execution and fading seeds,
   2-3 methods stacked (e.g. M, U and N_EDF). Slots where the outcomes diverge are
   highlighted.
5. **Failure inspector.** Each violated packet with its context: slack at arrival,
   SINR in its transmission slots, the interferers, whether its urgency flag was
   heard, and whether it was undeliverable alone.

## 4. Architecture

```
web/sim-viewer/
  README.md
  pyproject.toml       # own uv project
  server/
    app.py             # FastAPI routes; serves frontend/dist when built
    settings.py        # paths; finds the main checkout's outputs/ from a worktree
    registry.py        # experiments the app knows, with method descriptions
    catalog.py         # report gate, formal-output checks, per-deployment values
    exp011.py          # EXP-011 adapter: comparisons and tables of analysis.json
    geometry.py        # deployment geometry from Scenario.instance
    guards.py          # §2.1: Phase 1 geometry rule; allowlist from Phase 2
    trace.py           # (Phase 2) runs one episode, recording per-slot arrays
    jobs.py            # (Phase 2) worker pool, progress, disk cache
  tests/
  frontend/            # Vite + React + TypeScript
  .gitignore           # node_modules/, dist/, .cache/
```

### 4.1 Data sources

| Source | Content | Used by |
|---|---|---|
| `outputs/<exp>/eval/<label>/episodes.json` | `{spec: {N: [record]}}`; a record has instance, repeat, packets, violations, undeliverable, violation_rate, worst_violation_rate, feasible, infeasible_for_all, class_index, distance_m, shadowing_db, mean_power_mw, duty | explorer, map, exactness check |
| `outputs/<exp>/eval/<label>/summary.json` | protocol id, formal flag, provenance, config, split, screen, grid, per-policy summaries | explorer |
| `outputs/exp011/analysis.json` | the pre-registered analysis (P2, D_N, intervals) | explorer |
| `outputs/exp011/plan-exp011.json` | method groups, seed layers per N | grouping; descriptions come from design §3.1 via `registry.py` |
| `Scenario.instance(i)` | UE positions, traffic pattern (no policy run) | map |
| `trace.py` | per-slot arrays (§4.2) | views 3-5 |

The catalog starts with the EXP-011 layout. Other experiments' layouts are added
when they are needed.

### 4.2 Trace recording

`trace.py` runs the same loop as `capacity.run_capacity_episode`: the instance from
`Scenario.instance`, per-UE execution and fading seeds from `splits_v2`, and
`policy.configure(...)` for the semi-static references (N_CG). Policies come from
`rules_v2.make_policy_v2` with the same spec strings as the formal runs.

Recorded per slot, from `EpisodeStep` and the observations:

- `physical.actions` (subchannel, power id), `active`, `sinr_linear`,
  `service_bits`, `rssi_w`;
- `remaining_bits`, `completed`, `on_time`, `violations`, `released`;
- `own_gain_w` of every UE on every subchannel;
- for summary and yield policies, the urgency, summary and yield state, recomputed
  from the observations with the `dsa_period.urgency` functions.

Size: about 20,200 slots x 16 UEs x 12 fields, roughly 4 million values per episode.
Traces are cached as float32 `.npz` and served in windows of about 1,000 slots, plus
a downsampled overview. The cache key is commit + config hash + spec + N + split +
instance + repeat.

Cost: about 30-35 s per episode for a learned policy on one core (measured at about
600 slots/s). Replays run as background jobs with a progress indicator.

### 4.3 Exactness check

Every replay of an instance that has a formal record compares its per-UE violations
and packet counts with that record, as `diagnose_exp011 --check` does. The UI shows
"matches the formal run" or a red mismatch warning. A mismatch is a bug to fix, not
a result to show.

### 4.4 Workers

Workers are started with the `spawn` method, not `fork`. Any spec containing `marl:`
(including the wrapped `yield[...]@marl:` and `summary[...]@marl:` specs) runs with
`torch.set_num_threads(1)`. This avoids the OpenMP deadlock fixed in 4c8cc4f. The
pool size is configurable, 1-2 by default.

### 4.5 Frontend

- Canvas rendering for the timeline, the subchannel rows and the heatmap (SVG cannot
  handle 20k slots x 16 UEs).
- Hand-written SVG for the explorer charts (Phase 1); colours follow the dataviz
  reference palette, with methods A and B as slots 1 and 2, validated in light
  and dark mode.
- The view state (method(s), N, instance, slot range) lives in the URL, so a view
  can be bookmarked or shared.

## 5. Phases

Each phase ends with a review and a commit after approval.

| Phase | Deliverable | Changes `src/`? | Runs policies? |
|---|---|---|---|
| 0 | This plan and the empty directory | no | no |
| 1 | Results explorer and deployment map | no | no |
| 2 | Trace backend, cache, guards, exactness check, viewer slice registration | registration in `splits_v2` only | yes, 1-2 cores |
| 3 | Replay timeline and paired comparison | no | yes |
| 4 | Urgency/summary/yield overlays and failure inspector | no (possibly a trace hook later, §1) | yes |
| 5 (optional) | "What if" mode: move UEs or change θ, run on the viewer slice only, clearly marked exploratory | no | yes |

## 6. Tests

Backend (`pytest`, run in the app's own environment):

- the guards accept the §2.1 allowlist and refuse every held-out, screening,
  validation and reserve index;
- replays match the formal records for one rule-based policy (`best_local`), one
  `marl:` policy, one `yield[...]` policy and one `summary[...]` policy, on allowed
  instances;
- the recomputed urgency state matches the policy wrappers' own state on a short
  episode;
- the window and overview endpoints return the expected shapes and dtypes;
- the catalog reads the EXP-011 layout without writing to `outputs/`, and hides
  experiments whose report is not committed (§2.1).

Frontend:

- `vitest` for data transforms (windowing, divergence detection, colour scales);
- a manual check in the browser at each phase review.

Phase 1 has 37 backend tests: the report gate (shown, hidden before the report,
hidden for an uncommitted report); refusal of non-formal, dirty, wrong-set,
wrong-split and wrong-instance outputs; P2 and P1 recomputed from the episodes
equal `analysis.json` for every group and N; geometry is refused for v2-confirm1/2,
the EXP-011b sets, the screening sets and instances outside the shown records;
regenerated geometry equals the formal records (32 instances at each N) and is
nested across N; the API's status codes. There are 8 frontend tests (ECDF,
tallies, largest differences, nearest point, scales, formats).

## 7. Open items

- Whether `exp011b-stress` should stay refused permanently after the EXP-011b
  report, like the other selection sets (`exp008-screen256`, `exp009-screen256`).
- The `validation` split (used for the per-N calibration of the local rules and
  for N* evaluations) is not on the allowlist, so it is refused. Decide whether
  the app may replay it.
- Whether the catalog should cover experiments before EXP-011 (EXP-009/010, CG,
  CG-opt), and in which order.
- Whether a read-only trace hook in `src/` is worth adding after Phase 4.

## 8. Running

From `web/sim-viewer/` (in the worktree or, after a merge, in the main checkout):

```bash
uv sync                                  # once: the app's own .venv
(cd frontend && npm ci)                  # once: node_modules from package-lock.json

(cd frontend && npm run build)           # build the frontend into frontend/dist
uv run uvicorn server.app:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765/   (remote: ssh -L 8765:127.0.0.1:8765 <host>)
```

Development with hot reload: start the backend as above, then
`(cd frontend && npm run dev)` and open http://127.0.0.1:5173/ (it proxies `/api`
to port 8765).

Checks:

```bash
uv run pytest
uv run ruff check server tests && uv run ruff format --check server tests
(cd frontend && npm run typecheck && npm test)
```

Settings (environment variables): `SIM_VIEWER_MAIN` (the main checkout, found from
Git by default), `SIM_VIEWER_OUTPUTS` (default `<main>/outputs`),
`SIM_VIEWER_GATE_REF` (default `main`; a report must exist on this ref).

While a formal stage runs, start builds and tests with `nice -n 19`. Phase 1 runs no
simulation; the backend only reads JSON and regenerates geometry (about 2 ms per
deployment).

To add an experiment: add an entry to `server/registry.py` with its report path
and an adapter for its `analysis.json`, like `server/exp011.py`.
