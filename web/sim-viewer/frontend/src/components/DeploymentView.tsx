import { useState } from "react";

import type { Deployment } from "../api";
import { useWidth } from "../hooks";
import { pct } from "../lib/format";
import { linear } from "../lib/stats";

interface Props {
  deployment: Deployment;
  a: string;
  b: string;
}

const SHAPES = ["circle", "square", "triangle"] as const;

function Marker({ shape, cx, cy }: { shape: (typeof SHAPES)[number]; cx: number; cy: number }) {
  const style = { fill: "var(--ink-2)", stroke: "var(--surface)", strokeWidth: 2 };
  if (shape === "circle") return <circle cx={cx} cy={cy} r={5} {...style} />;
  if (shape === "square") return <rect x={cx - 4.5} y={cy - 4.5} width={9} height={9} rx={1} {...style} />;
  return <path d={`M${cx},${cy - 6}L${cx + 5.5},${cy + 4}L${cx - 5.5},${cy + 4}Z`} {...style} />;
}

function argmax(values: number[]): number {
  let best = -1;
  values.forEach((v, i) => {
    if (v > 0 && (best < 0 || v > values[best])) best = i;
  });
  return best;
}

/** AP, coverage disk and UEs (shape = traffic class); rings mark each method's worst UE. */
function DeploymentMap({ deployment, a, b, hovered, setHovered }: Props & {
  hovered: number;
  setHovered: (i: number) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const size = Math.max(240, Math.min(width, 420));
  const { radius_m: radius, positions_m: positions, class_index: classes, class_names: names } =
    deployment.geometry;
  const margin = 18;
  const x = linear(-radius, radius, margin, size - margin);
  const y = linear(-radius, radius, size - margin, margin);
  const r = (meters: number) => x(meters) - x(0);
  const rateA = deployment.groups[a].rate_mean;
  const rateB = deployment.groups[b].rate_mean;
  const worstA = argmax(rateA);
  const worstB = argmax(rateB);
  return (
    <div>
      <div className="legend">
        {names.map((name, k) => (
          <span key={name}>
            <svg width={12} height={12} style={{ verticalAlign: "middle" }}>
              <Marker shape={SHAPES[k]} cx={6} cy={6.5} />
            </svg>{" "}
            {name} (every {deployment.geometry.period_slots[classes.indexOf(k)] ?? "—"} slots)
          </span>
        ))}
        <span>
          <span className="swatch" style={{ border: "2px solid var(--series-a)", borderRadius: 6 }} /> worst
          UE of A · {a}
        </span>
        <span>
          <span className="swatch" style={{ border: "2px solid var(--series-b)", borderRadius: 6 }} /> worst
          UE of B · {b}
        </span>
      </div>
      <div ref={ref} className="chart">
        <svg width={size} height={size} role="img" aria-label={`Deployment ${deployment.instance} map`}>
          <circle cx={x(0)} cy={y(0)} r={r(radius)} fill="none" stroke="var(--axis)" strokeWidth={1} />
          <circle cx={x(0)} cy={y(0)} r={r(radius / 2)} fill="none" stroke="var(--grid)" strokeWidth={1} />
          <text x={x(0) + r(radius / 2) + 4} y={y(0) - 4} fontSize={10} fill="var(--muted)">
            {radius / 2} m
          </text>
          <text x={x(0) + r(radius) - 2} y={y(0) - 4} fontSize={10} fill="var(--muted)" textAnchor="end">
            {radius} m
          </text>
          <rect x={x(0) - 5} y={y(0) - 5} width={10} height={10} fill="var(--ink)" />
          <text x={x(0) + 8} y={y(0) + 14} fontSize={11} fill="var(--ink-2)">
            AP
          </text>
          {positions.map(([px, py], i) => (
            <g
              key={i}
              onPointerEnter={() => setHovered(i)}
              onPointerLeave={() => setHovered(-1)}
              style={{ cursor: "default" }}
            >
              <circle cx={x(px)} cy={y(py)} r={12} fill="transparent" />
              {i === worstB && (
                <circle cx={x(px)} cy={y(py)} r={12} fill="none" stroke="var(--series-b)" strokeWidth={2} />
              )}
              {i === worstA && (
                <circle cx={x(px)} cy={y(py)} r={9} fill="none" stroke="var(--series-a)" strokeWidth={2} />
              )}
              {i === hovered && (
                <circle cx={x(px)} cy={y(py)} r={15} fill="none" stroke="var(--ink)" strokeWidth={1} />
              )}
              <Marker shape={SHAPES[classes[i]]} cx={x(px)} cy={y(py)} />
              <text x={x(px) + 9} y={y(py) - 8} fontSize={10} fill="var(--muted)">
                {i}
              </text>
            </g>
          ))}
        </svg>
        {hovered >= 0 && (
          <div
            className="tooltip"
            style={{
              left: Math.min(x(positions[hovered][0]) + 16, size - 170),
              top: Math.max(y(positions[hovered][1]) - 70, 0),
            }}
          >
            <div className="label">
              UE {hovered} · {names[classes[hovered]]} · {deployment.geometry.distance_m[hovered].toFixed(1)} m ·
              shadowing {deployment.geometry.shadowing_db[hovered].toFixed(1)} dB
            </div>
            <div>
              <span className="line-key" style={{ background: "var(--series-a)" }} />{" "}
              <span className="value">{pct(rateA[hovered])}</span> <span className="label">A · {a}</span>
            </div>
            <div>
              <span className="line-key" style={{ background: "var(--series-b)" }} />{" "}
              <span className="value">{pct(rateB[hovered])}</span> <span className="label">B · {b}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function RateBar({ value, max, color, title }: { value: number; max: number; color: string; title: string }) {
  return (
    <div className="bar-cell" title={title}>
      {pct(value)}
      <div className="bar-track">
        {value > 0 && <div className="bar-fill" style={{ width: `${(value / max) * 100}%`, background: color }} />}
      </div>
    </div>
  );
}

/** Per-UE violation rates (seed-layer means) next to the deployment's records. */
function UeTable({ deployment, a, b, hovered, setHovered }: Props & {
  hovered: number;
  setHovered: (i: number) => void;
}) {
  const g = deployment.geometry;
  const rateA = deployment.groups[a].rate_mean;
  const rateB = deployment.groups[b].rate_mean;
  const max = Math.max(...rateA, ...rateB, 1e-12);
  const perSeed = (group: string, i: number) =>
    "per seed layer: " + deployment.groups[group].layers.map((layer) => pct(layer.violation_rate[i])).join(", ");
  const undeliverable = deployment.groups[a].layers[0].undeliverable;
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>UE</th>
            <th>Class</th>
            <th className="num">Distance</th>
            <th className="num">Shadowing</th>
            <th className="num">Packets</th>
            <th className="num">
              <span className="swatch" style={{ background: "var(--series-a)" }} /> A · {a}
            </th>
            <th className="num">
              <span className="swatch" style={{ background: "var(--series-b)" }} /> B · {b}
            </th>
            <th className="num" title="packets no transmission could deliver, even alone">
              Undeliverable
            </th>
          </tr>
        </thead>
        <tbody>
          {g.distance_m.map((distance, i) => (
            <tr
              key={i}
              className={i === hovered ? "current-col" : ""}
              onPointerEnter={() => setHovered(i)}
              onPointerLeave={() => setHovered(-1)}
            >
              <td>{i}</td>
              <td>{g.class_names[g.class_index[i]]}</td>
              <td className="num">{distance.toFixed(1)} m</td>
              <td className="num">{g.shadowing_db[i].toFixed(1)} dB</td>
              <td className="num">{deployment.records.packets[i]}</td>
              <td className="num">
                <RateBar value={rateA[i]} max={max} color="var(--series-a)" title={perSeed(a, i)} />
              </td>
              <td className="num">
                <RateBar value={rateB[i]} max={max} color="var(--series-b)" title={perSeed(b, i)} />
              </td>
              <td className="num">{undeliverable[i]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DeploymentView(props: Props) {
  const [hovered, setHovered] = useState(-1);
  return (
    <div className="deploy-grid">
      <DeploymentMap {...props} hovered={hovered} setHovered={setHovered} />
      <UeTable {...props} hovered={hovered} setHovered={setHovered} />
    </div>
  );
}
