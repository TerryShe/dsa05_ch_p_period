import type { Overview } from "../api";
import { pct, share } from "../lib/format";

export type Metric = "P2" | "P1" | "p90_w" | "all_ue_mean_rate";

export const METRICS: Record<Metric, string> = {
  P2: "P2: mean worst-UE violation rate",
  P1: "P1: share of deployments feasible at ε = 0.01",
  p90_w: "p90 of the worst-UE violation rate",
  all_ue_mean_rate: "all-UE mean violation rate (guard)",
};

interface Props {
  overview: Overview;
  metric: Metric;
  n: number;
  a: string;
  b: string;
}

/** Method x N table of the analysis' point estimates (no per-method intervals exist). */
export function ResultsTable({ overview, metric, n, a, b }: Props) {
  const grid = [...new Set(overview.groups.flatMap((g) => g.grid))].sort((p, q) => p - q);
  const values = overview.tables[metric];
  const format = metric === "P1" ? share : pct;
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Description</th>
            {grid.map((column) => (
              <th key={column} className={`num ${column === n ? "current-col" : ""}`}>
                N = {column}
              </th>
            ))}
            {metric === "P1" && <th className="num">N* on grid</th>}
          </tr>
        </thead>
        <tbody>
          {overview.groups.map((group) => {
            const nStar = overview.tables.N_star_on_grid[group.name];
            return (
              <tr key={group.name}>
                <td>
                  {group.name}
                  {group.name === a && (
                    <span className="tag">
                      <span className="swatch" style={{ background: "var(--series-a)" }} />A
                    </span>
                  )}
                  {group.name === b && (
                    <span className="tag">
                      <span className="swatch" style={{ background: "var(--series-b)" }} />B
                    </span>
                  )}
                </td>
                <td className="desc">
                  {group.description}
                  {group.layers > 1 && ` · ${group.layers} seed layers`}
                </td>
                {grid.map((column) => {
                  const value = values[group.name]?.[String(column)];
                  return (
                    <td key={column} className={`num ${column === n ? "current-col" : ""}`}>
                      {value === undefined ? "—" : format(value)}
                    </td>
                  );
                })}
                {metric === "P1" && (
                  <td className="num">
                    {nStar === undefined ? "—" : nStar === null ? `< ${grid[0]}` : nStar}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
