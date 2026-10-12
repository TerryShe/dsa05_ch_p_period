import type { ComparisonRow, Family } from "../api";
import { pp, ratio, signedFixed } from "../lib/format";
import { linear } from "../lib/stats";

/** Where a 95% interval lies relative to zero. */
export function excludesZero([lo, hi]: [number, number]): "below" | "above" | null {
  if (hi < 0) return "below";
  if (lo > 0) return "above";
  return null;
}

/** Interval glyph on the row's own scale, with zero marked (rows differ by orders of magnitude). */
function Glyph({ row }: { row: ComparisonRow }) {
  const width = 140;
  const height = 16;
  const [lo, hi] = row.ci95;
  const left = Math.min(lo, 0);
  const right = Math.max(hi, 0);
  const pad = (right - left) * 0.08 || 1;
  const x = linear(left - pad, right + pad, 4, width - 4);
  const mid = height / 2;
  return (
    <svg width={width} height={height} role="img" aria-label={`interval ${lo} to ${hi}`}>
      <line x1={x(0)} x2={x(0)} y1={1} y2={height - 1} stroke="var(--axis)" strokeWidth={1} />
      <line
        x1={x(lo)}
        x2={x(hi)}
        y1={mid}
        y2={mid}
        stroke="var(--ink-2)"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <circle cx={x(row.difference)} cy={mid} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
    </svg>
  );
}

function FamilyTable({ family }: { family: Family }) {
  const format = family.scale === "rate" ? pp : (v: number) => signedFixed(v, 4);
  const unit = family.scale === "rate" ? " pp" : "";
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Row</th>
            <th className="num">Difference</th>
            <th className="num">95% interval</th>
            <th className="num">Ratio of means</th>
            <th>Interval, zero marked</th>
            <th>Excludes 0</th>
          </tr>
        </thead>
        <tbody>
          {family.rows.map((row) => {
            const side = excludesZero(row.ci95);
            const core = family.core?.includes(row.key);
            return (
              <tr key={row.key}>
                <td>
                  {row.label}
                  {core && <span className="tag">core</span>}
                  {row.deployments !== null && (
                    <span className="tag">{row.deployments} deployments</span>
                  )}
                </td>
                <td className="num">
                  {format(row.difference)}
                  {unit}
                </td>
                <td className="num">
                  [{format(row.ci95[0])}, {format(row.ci95[1])}]{unit}
                </td>
                <td className="num">{ratio(row.ratio)}</td>
                <td>
                  <Glyph row={row} />
                </td>
                <td>{side === null ? "no" : `yes, ${side} 0`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The pre-registered comparisons of analysis.json, as written; the main rule opens first. */
export function Comparisons({ families }: { families: Family[] }) {
  return (
    <div>
      {families.map((family) => (
        <details key={family.id} className="family" open={family.id === "D_N"}>
          <summary>
            {family.title}
            {family.note && <span className="note">{family.note}</span>}
          </summary>
          <FamilyTable family={family} />
        </details>
      ))}
    </div>
  );
}
