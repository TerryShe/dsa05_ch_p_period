import { useMemo, useState } from "react";

import type { Paired } from "../api";
import { useWidth } from "../hooks";
import { pct, share } from "../lib/format";
import { DELTA, ecdf, linear, logRate, shareAt } from "../lib/stats";
import { LOG_MAX, LOG_MIN, RateGrid } from "./rateAxis";

const HEIGHT = 280;
const MARGIN = { left: 48, right: 16, top: 12, bottom: 44 };

function stepPath(steps: { x: number[]; y: number[] }, x: (v: number) => number, y: (v: number) => number) {
  let d = `M${x(LOG_MIN)},${y(0)}`;
  steps.x.forEach((value, i) => {
    d += `H${x(logRate(value))}V${y(steps.y[i])}`;
  });
  return `${d}H${x(LOG_MAX)}`;
}

/** Share of deployments whose worst-UE violation rate is at most w, for A and B. */
export function Ecdf({ paired }: { paired: Paired }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [pointer, setPointer] = useState<number | null>(null);
  const w = Math.max(280, width);
  const x = linear(LOG_MIN, LOG_MAX, MARGIN.left, w - MARGIN.right);
  const y = linear(0, 1, HEIGHT - MARGIN.bottom, MARGIN.top);
  const stepsA = useMemo(() => ecdf(paired.a.w), [paired.a.w]);
  const stepsB = useMemo(() => ecdf(paired.b.w), [paired.b.w]);
  const at = pointer === null ? null : Math.max(0, 10 ** pointer - DELTA);

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - box.left;
    if (px < MARGIN.left || px > w - MARGIN.right) return setPointer(null);
    setPointer(LOG_MIN + ((px - MARGIN.left) / (w - MARGIN.left - MARGIN.right)) * (LOG_MAX - LOG_MIN));
  };

  return (
    <div>
      <div className="legend">
        <span>
          <span className="line-key" style={{ background: "var(--series-a)" }} /> A · {paired.a.group} (
          {share(shareAt(stepsA, 0))} with w = 0)
        </span>
        <span>
          <span className="line-key" style={{ background: "var(--series-b)" }} /> B · {paired.b.group} (
          {share(shareAt(stepsB, 0))} with w = 0)
        </span>
      </div>
      <div ref={ref} className="chart">
        <svg
          width={w}
          height={HEIGHT}
          onPointerMove={onMove}
          onPointerLeave={() => setPointer(null)}
          role="img"
          aria-label="Distribution of the worst-UE violation rate over deployments"
        >
          <RateGrid scale={x} from={MARGIN.top} to={HEIGHT - MARGIN.bottom} orientation="bottom" />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={MARGIN.left} x2={w - MARGIN.right} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth={1} />
              <text x={MARGIN.left - 6} y={y(v) + 4} textAnchor="end" fontSize={11} fill="var(--muted)">
                {v * 100}%
              </text>
            </g>
          ))}
          <path d={stepPath(stepsB, x, y)} fill="none" stroke="var(--series-b)" strokeWidth={2} strokeLinejoin="round" />
          <path d={stepPath(stepsA, x, y)} fill="none" stroke="var(--series-a)" strokeWidth={2} strokeLinejoin="round" />
          {pointer !== null && (
            <line x1={x(pointer)} x2={x(pointer)} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} stroke="var(--ink-2)" strokeWidth={1} />
          )}
          <text x={(MARGIN.left + w - MARGIN.right) / 2} y={HEIGHT - 8} textAnchor="middle" fontSize={12} fill="var(--ink-2)">
            worst-UE violation rate w (at most)
          </text>
        </svg>
        {pointer !== null && at !== null && (
          <div className="tooltip" style={{ left: Math.min(x(pointer) + 12, w - 170), top: MARGIN.top }}>
            <div className="label">w ≤ {pct(at)}</div>
            <div>
              <span className="line-key" style={{ background: "var(--series-a)" }} />{" "}
              <span className="value">{share(shareAt(stepsA, at))}</span>{" "}
              <span className="label">A · {paired.a.group}</span>
            </div>
            <div>
              <span className="line-key" style={{ background: "var(--series-b)" }} />{" "}
              <span className="value">{share(shareAt(stepsB, at))}</span>{" "}
              <span className="label">B · {paired.b.group}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
