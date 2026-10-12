import { useState } from "react";

import type { Paired } from "../api";
import { useWidth } from "../hooks";
import { pct } from "../lib/format";
import { linear, logRate, nearest, tally } from "../lib/stats";
import { LOG_MAX, LOG_MIN, RateGrid } from "./rateAxis";

interface Props {
  paired: Paired;
  selected: number | null;
  onSelect: (instance: number) => void;
}

const MARGIN = { left: 52, right: 12, top: 12, bottom: 44 };

/** Worst-UE violation rate per deployment, A (y) against B (x); one dot per deployment. */
export function PairedScatter({ paired, selected, onSelect }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hovered, setHovered] = useState(-1);
  const size = Math.max(240, Math.min(width, 460));
  const plot = size - MARGIN.left - MARGIN.right;
  const x = linear(LOG_MIN, LOG_MAX, MARGIN.left, MARGIN.left + plot);
  const y = linear(LOG_MIN, LOG_MAX, MARGIN.top + plot, MARGIN.top);
  const { a, b, instances } = paired;
  const xs = b.w.map((v) => x(logRate(v)));
  const ys = a.w.map((v) => y(logRate(v)));
  const counts = tally(a.w, b.w);
  const fill = (i: number) =>
    a.w[i] < b.w[i] ? "var(--series-a)" : a.w[i] > b.w[i] ? "var(--series-b)" : "var(--muted)";
  // Equal pairs first, so the deployments that differ stay on top.
  const order = instances.map((_, i) => i).sort((p, q) => Number(a.w[p] !== b.w[p]) - Number(a.w[q] !== b.w[q]));
  const selectedIndex = selected === null ? -1 : instances.indexOf(selected);

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    setHovered(nearest(xs, ys, event.clientX - box.left, event.clientY - box.top, 24));
  };

  return (
    <div>
      <div className="legend">
        <span>
          <span className="swatch" style={{ background: "var(--series-a)" }} /> A lower ({a.group}):{" "}
          {counts.aLower}
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--series-b)" }} /> B lower ({b.group}):{" "}
          {counts.bLower}
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--muted)" }} /> equal: {counts.equal} (
          {counts.bothZero} both 0)
        </span>
      </div>
      <div ref={ref} className="chart">
        <svg
          width={size}
          height={size + MARGIN.bottom - MARGIN.right}
          onPointerMove={onMove}
          onPointerLeave={() => setHovered(-1)}
          onClick={() => hovered >= 0 && onSelect(instances[hovered])}
          style={{ cursor: hovered >= 0 ? "pointer" : "default" }}
          role="img"
          aria-label={`Worst-UE violation rate per deployment, ${a.group} against ${b.group}`}
        >
          <RateGrid scale={x} from={MARGIN.top} to={MARGIN.top + plot} orientation="bottom" />
          <RateGrid scale={y} from={MARGIN.left} to={MARGIN.left + plot} orientation="left" />
          <line
            x1={x(LOG_MIN)}
            y1={y(LOG_MIN)}
            x2={x(LOG_MAX)}
            y2={y(LOG_MAX)}
            stroke="var(--axis)"
            strokeWidth={1}
          />
          <text x={MARGIN.left + plot - 4} y={MARGIN.top + plot - 8} textAnchor="end" fontSize={11} fill="var(--muted)">
            A lower
          </text>
          <text x={MARGIN.left + 8} y={MARGIN.top + 14} fontSize={11} fill="var(--muted)">
            B lower
          </text>
          {order.map((i) => (
            <circle
              key={instances[i]}
              cx={xs[i]}
              cy={ys[i]}
              r={4}
              fill={fill(i)}
              stroke="var(--surface)"
              strokeWidth={2}
            />
          ))}
          {counts.bothZero > 0 && (
            <text x={x(LOG_MIN) + 10} y={y(LOG_MIN) - 8} fontSize={11} fill="var(--ink-2)">
              {counts.bothZero} at 0 / 0
            </text>
          )}
          {selectedIndex >= 0 && (
            <circle cx={xs[selectedIndex]} cy={ys[selectedIndex]} r={8} fill="none" stroke="var(--ink)" strokeWidth={2} />
          )}
          {hovered >= 0 && (
            <circle cx={xs[hovered]} cy={ys[hovered]} r={7} fill="none" stroke="var(--ink-2)" strokeWidth={1} />
          )}
          <text x={MARGIN.left + plot / 2} y={MARGIN.top + plot + 36} textAnchor="middle" fontSize={12} fill="var(--ink-2)">
            B · {b.group}: worst-UE violation rate
          </text>
          <text
            transform={`translate(12 ${MARGIN.top + plot / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fill="var(--ink-2)"
          >
            A · {a.group}: worst-UE violation rate
          </text>
        </svg>
        {hovered >= 0 && (
          <div
            className="tooltip"
            style={{ left: Math.min(xs[hovered] + 12, size - 150), top: Math.max(ys[hovered] - 56, 0) }}
          >
            <div className="label">Deployment {instances[hovered]} · click to open</div>
            <div>
              <span className="line-key" style={{ background: "var(--series-a)" }} />{" "}
              <span className="value">{pct(a.w[hovered])}</span> <span className="label">A · {a.group}</span>
            </div>
            <div>
              <span className="line-key" style={{ background: "var(--series-b)" }} />{" "}
              <span className="value">{pct(b.w[hovered])}</span> <span className="label">B · {b.group}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
