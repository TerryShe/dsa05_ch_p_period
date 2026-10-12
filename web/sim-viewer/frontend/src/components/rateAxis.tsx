import { logRate } from "../lib/stats";

// Violation rates on log10(w + delta): zero sits at the left end (10^-4), 100% at the right.
export const LOG_MIN = logRate(0);
export const LOG_MAX = logRate(1);
export const RATE_TICKS: [number, string][] = [
  [0, "0"],
  [0.001, "0.1%"],
  [0.01, "1%"],
  [0.1, "10%"],
  [1, "100%"],
];

interface AxisProps {
  scale: (v: number) => number;
  from: number;
  to: number;
  orientation: "bottom" | "left";
}

/** Hairline gridlines and muted tick labels for a rate axis. */
export function RateGrid({ scale, from, to, orientation }: AxisProps) {
  return (
    <g>
      {RATE_TICKS.map(([value, label]) => {
        const p = scale(logRate(value));
        return orientation === "bottom" ? (
          <g key={label}>
            <line x1={p} x2={p} y1={from} y2={to} stroke="var(--grid)" strokeWidth={1} />
            <text x={p} y={to + 16} textAnchor="middle" fontSize={11} fill="var(--muted)">
              {label}
            </text>
          </g>
        ) : (
          <g key={label}>
            <line x1={from} x2={to} y1={p} y2={p} stroke="var(--grid)" strokeWidth={1} />
            <text x={from - 6} y={p + 4} textAnchor="end" fontSize={11} fill="var(--muted)">
              {label}
            </text>
          </g>
        );
      })}
    </g>
  );
}
