import { describe, expect, it } from "vitest";

import { excludesZero } from "../components/Comparisons";
import { pct, pp, ratio, share } from "./format";
import { DELTA, ecdf, largestDifferences, linear, logRate, nearest, shareAt, tally } from "./stats";

describe("ecdf", () => {
  it("collapses ties into one step with the cumulative share", () => {
    const steps = ecdf([0, 0, 0.01, 0.002]);
    expect(steps.x).toEqual([0, 0.002, 0.01]);
    expect(steps.y).toEqual([0.5, 0.75, 1]);
  });

  it("reads the step function at any x", () => {
    const steps = ecdf([0, 0, 0.01, 0.002]);
    expect(shareAt(steps, -1)).toBe(0);
    expect(shareAt(steps, 0)).toBe(0.5);
    expect(shareAt(steps, 0.005)).toBe(0.75);
    expect(shareAt(steps, 1)).toBe(1);
  });
});

describe("paired helpers", () => {
  it("tallies which side is lower, with ties at zero", () => {
    expect(tally([0, 0.1, 0.2, 0], [0, 0.2, 0.1, 0.3])).toEqual({
      aLower: 2,
      bLower: 1,
      equal: 1,
      bothZero: 1,
    });
  });

  it("orders the largest differences by magnitude and drops equal pairs", () => {
    const top = largestDifferences([10, 11, 12, 13], [0.1, 0, 0.5, 0.2], [0.1, 0.3, 0.1, 0.1], 2);
    expect(top.map((d) => d.instance)).toEqual([12, 11]);
    expect(top[1].diff).toBeCloseTo(-0.3);
  });

  it("finds the nearest point within the radius only", () => {
    expect(nearest([0, 10, 30], [0, 0, 0], 12, 1, 24)).toBe(1);
    expect(nearest([0, 10, 30], [0, 0, 0], 100, 100, 24)).toBe(-1);
  });
});

describe("scales", () => {
  it("puts zero at log10(delta) and maps linearly", () => {
    expect(logRate(0)).toBeCloseTo(Math.log10(DELTA));
    const x = linear(-4, 0, 0, 400);
    expect(x(-2)).toBe(200);
  });
});

describe("formats", () => {
  it("formats rates, shares, differences and ratios", () => {
    expect(pct(0)).toBe("0%");
    expect(pct(0.002778645833)).toBe("0.278%");
    expect(pct(0.0746)).toBe("7.46%");
    expect(share(0.955078125)).toBe("95.5%");
    expect(pp(-0.019156901)).toBe("−1.916");
    expect(pp(0.00018847)).toBe("+0.019");
    expect(ratio(null)).toBe("—");
    expect(ratio(0.7957)).toBe("×0.80");
  });

  it("classifies intervals against zero", () => {
    expect(excludesZero([-0.003, -0.001])).toBe("below");
    expect(excludesZero([0.001, 0.002])).toBe("above");
    expect(excludesZero([-0.001, 0.002])).toBeNull();
  });
});
