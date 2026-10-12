import { useEffect } from "react";

import { api, type Overview } from "./api";
import { Comparisons } from "./components/Comparisons";
import { DeploymentView } from "./components/DeploymentView";
import { Ecdf } from "./components/Ecdf";
import { PairedScatter } from "./components/PairedScatter";
import { METRICS, type Metric, ResultsTable } from "./components/ResultsTable";
import { useFetch, useUrlState } from "./hooks";
import { pct, pp, share } from "./lib/format";
import { largestDifferences } from "./lib/stats";

function commonGrid(overview: Overview, a: string, b: string): number[] {
  const grid = (name: string) => overview.groups.find((g) => g.name === name)?.grid ?? [];
  return grid(a).filter((n) => grid(b).includes(n));
}

export function App() {
  const [state, update] = useUrlState({ exp: "", metric: "P2", n: "8", a: "U", b: "M", inst: "" });
  const listing = useFetch("experiments", api.experiments);
  const shown = listing.data?.filter((e) => e.shown) ?? [];
  const hidden = listing.data?.filter((e) => !e.shown) ?? [];
  const exp = state.exp || shown[0]?.id || "";

  const overview = useFetch(exp ? `overview:${exp}` : null, () => api.overview(exp));
  const grid = overview.data ? commonGrid(overview.data, state.a, state.b) : [];
  const n = Number(state.n);
  const valid = overview.data !== null && grid.includes(n) && !overview.stale;

  // Keep N on the grid that A and B share.
  useEffect(() => {
    if (overview.data && grid.length && !grid.includes(n)) {
      update({ n: String(grid.includes(8) ? 8 : grid[0]), inst: "" });
    }
  }, [overview.data, grid, n, update]);

  const paired = useFetch(valid ? `paired:${exp}:${state.a}:${state.b}:${n}` : null, () =>
    api.paired(exp, state.a, state.b, n),
  );
  const instance = state.inst === "" ? null : Number(state.inst);
  const deployment = useFetch(
    valid && instance !== null ? `deployment:${exp}:${instance}:${n}:${state.a}:${state.b}` : null,
    () => api.deployment(exp, instance as number, n, [state.a, state.b]),
  );

  if (listing.error) return <div className="page error">Could not reach the backend: {listing.error}</div>;
  if (listing.data && !shown.length) {
    return <div className="page">No experiment has a committed report yet, so there is nothing to show.</div>;
  }
  const o = overview.data;
  const p = paired.data;
  const matches = (side: { P2: number; P2_analysis: number }) =>
    Math.abs(side.P2 - side.P2_analysis) <= 1e-12 * Math.max(1, Math.abs(side.P2_analysis));
  const top = p ? largestDifferences(p.instances, p.a.w, p.b.w, 12) : [];

  return (
    <div className="page">
      <header>
        <h1>{o?.title ?? "sim-viewer"}</h1>
        {o && (
          <div className="meta">
            <span>
              {o.instances} deployments of <code>{o.evaluation_set}</code>
            </span>
            <span>
              report <code>{o.report}</code>
            </span>
            <span>
              formal runs at {o.commits.map((c) => c.slice(0, 7)).join(", ")}
            </span>
            <span>main rule: {o.decision}</span>
          </div>
        )}
        {hidden.length > 0 && (
          <div className="meta">
            Not shown until their report is committed: {hidden.map((e) => e.id).join(", ")}
          </div>
        )}
      </header>

      <div className="filters">
        <label>
          Experiment
          <select value={exp} onChange={(e) => update({ exp: e.target.value, inst: "" })}>
            {shown.map((e) => (
              <option key={e.id} value={e.id}>
                {e.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Table metric
          <select value={state.metric} onChange={(e) => update({ metric: e.target.value })}>
            {Object.entries(METRICS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Method A
          <select value={state.a} onChange={(e) => update({ a: e.target.value })}>
            {o?.groups.map((g) => (
              <option key={g.name} value={g.name}>
                {g.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Method B
          <select value={state.b} onChange={(e) => update({ b: e.target.value })}>
            {o?.groups.map((g) => (
              <option key={g.name} value={g.name}>
                {g.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          N (UEs)
          <select value={state.n} onChange={(e) => update({ n: e.target.value, inst: "" })}>
            {grid.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>

      {overview.error && <p className="error">{overview.error}</p>}
      {o && (
        <section className={`card ${overview.stale ? "stale" : ""}`}>
          <h2>Results by method and N</h2>
          <p className="sub">
            {METRICS[state.metric as Metric]}. Point estimates from <code>analysis.json</code>; learned
            methods average their seed layers. Intervals exist only for the paired differences below.
          </p>
          <ResultsTable overview={o} metric={state.metric as Metric} n={n} a={state.a} b={state.b} />
        </section>
      )}

      {o && (
        <section className="card">
          <h2>Pre-registered comparisons</h2>
          <p className="sub">
            Paired differences of P2 (or of the named metric) with their bootstrap 95% intervals, exactly as
            written by the analysis. Each glyph uses its row's own scale.
          </p>
          <Comparisons families={o.comparisons} />
        </section>
      )}

      {paired.error && <p className="error">{paired.error}</p>}
      {p && (
        <section className={`card ${paired.stale ? "stale" : ""}`}>
          <h2>
            Deployments at N = {p.n}: A · {p.a.group} against B · {p.b.group}
          </h2>
          <div className="tally">
            {[p.a, p.b].map((side, k) => (
              <span key={k}>
                <span className="swatch" style={{ background: k ? "var(--series-b)" : "var(--series-a)" }} />{" "}
                {k ? "B" : "A"} · {side.group}: P2 {pct(side.P2)}, P1 {share(side.P1)}
                {side.layers > 1 && `, ${side.layers} seed layers`}{" "}
                {matches(side) ? (
                  <span className="check-ok">✓ matches analysis.json</span>
                ) : (
                  <span className="check-bad">✕ differs from analysis.json ({pct(side.P2_analysis)})</span>
                )}
              </span>
            ))}
          </div>
          <div className="two-col">
            <PairedScatter
              paired={p}
              selected={instance}
              onSelect={(i) => update({ inst: String(i) })}
            />
            <Ecdf paired={p} />
          </div>
          <h2 style={{ marginTop: 16 }}>Largest differences</h2>
          <p className="sub">
            Deployments where A and B differ most in the worst-UE violation rate (seed-layer means).
          </p>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Deployment</th>
                  <th className="num">A · {p.a.group}</th>
                  <th className="num">B · {p.b.group}</th>
                  <th className="num">A − B</th>
                </tr>
              </thead>
              <tbody>
                {top.map((d) => (
                  <tr key={d.instance} className={d.instance === instance ? "current-col" : ""}>
                    <td>
                      <button className="link" onClick={() => update({ inst: String(d.instance) })}>
                        {d.instance}
                      </button>
                    </td>
                    <td className="num">{pct(d.a)}</td>
                    <td className="num">{pct(d.b)}</td>
                    <td className="num">{pp(d.diff)} pp</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {deployment.error && <p className="error">{deployment.error}</p>}
      {deployment.data && instance !== null && (
        <section className={`card ${deployment.stale ? "stale" : ""}`}>
          <h2>
            Deployment {deployment.data.instance} at N = {deployment.data.n}
          </h2>
          <p className="sub">
            Worst-UE violation rate: A · {state.a} {pct(deployment.data.groups[state.a]?.w ?? 0)}, B · {state.b}{" "}
            {pct(deployment.data.groups[state.b]?.w ?? 0)}. Geometry regenerated from the scenario file and
            checked against the formal record.
          </p>
          {deployment.data.groups[state.a] && deployment.data.groups[state.b] && (
            <DeploymentView deployment={deployment.data} a={state.a} b={state.b} />
          )}
        </section>
      )}
    </div>
  );
}
