// Number formats. Violation rates read as percentages; differences of rates as
// percentage points (pp).

const MINUS = "−";

function signed(text: string, value: number): string {
  if (value > 0) return `+${text}`;
  return text.replace("-", MINUS);
}

/** A rate in [0, 1] as a percentage with three significant digits ("0" stays "0%"). */
export function pct(value: number): string {
  if (value === 0) return "0%";
  const text = (value * 100).toPrecision(3);
  return `${Number(text)}%`;
}

/** A share in [0, 1] as a percentage with one decimal. */
export function share(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** A difference of rates in signed percentage points, three decimals. */
export function pp(value: number): string {
  return signed((value * 100).toFixed(3), value);
}

/** A plain signed number with fixed decimals. */
export function signedFixed(value: number, digits = 3): string {
  return signed(value.toFixed(digits), value);
}

export function ratio(value: number | null): string {
  return value === null ? "—" : `×${value.toFixed(2)}`;
}
