"""Experiments the viewer knows (README §4.1).

An entry is shown only when its report exists on the gate ref (README §2.1);
the catalog starts with the EXP-011 layout, other layouts are added when needed.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    outputs: str  # directory under the outputs root
    report: str  # path in the repository; must exist on the gate ref
    evaluation_set: str  # registered splits_v2 screening set the evaluations read
    plan: str  # file under outputs/<outputs>: {"groups": ..., "grids": ...}
    analysis: str  # file under outputs/<outputs>: the pre-registered analysis
    config: str  # scenario file, relative to the repository
    descriptions: dict[str, str] = field(default_factory=dict)


EXP011 = Experiment(
    id="exp011",
    title="EXP-DSA05-011: urgency state summary",
    outputs="exp011",
    report="docs/results/exp011_20261012/report.md",
    evaluation_set="exp011-eval512",
    plan="plan-exp011.json",
    analysis="analysis.json",
    config="configs/scenario-v2.toml",
    # Design §3.1 (docs/experiment-designs/011-urgency-information-coordination_20261010.md).
    descriptions={
        "U": "L0 + L1 urgency summary, learned response (θ = −10 dB)",
        "M": "L0, learned (EXP-009 S1), checkpoints re-selected by P2",
        "M+Y": "M + hand yield rule: a non-urgent UE defers if its subchannel is flagged (θ = −10 dB)",
        "O": "L0 + L1 occupancy summary, learned",
        "L": "L0 + learned 1-bit message, learned",
        "U-hide-all": "U with the broadcast hidden from every UE",
        "U-hide-nonurgent": "U with the broadcast hidden from non-urgent UEs",
        "U-hide-urgent": "U with the broadcast hidden from urgent UEs",
        "best_local": "best calibrated local rule (L0)",
        "best_local+Y": "best_local + yield rule",
        "CG-opt": "optimized configured grant (L2, static exclusive)",
        "N_EDF": "central EDF scheduler (L∞ reference)",
        "M_exp009": "M with EXP-009's original checkpoint selection",
        "U0": "U trained and evaluated at θ = 0 dB",
        "U@all": "U, every flag read",
        "U@-15": "U at θ = −15 dB",
        "U@-5": "U at θ = −5 dB",
        "U@0": "U at θ = 0 dB",
        "M+Y@all": "M+Y, every flag read",
        "M+Y@-15": "M+Y at θ = −15 dB",
        "M+Y@-5": "M+Y at θ = −5 dB",
        "M+Y@0": "M+Y at θ = 0 dB",
    },
)

EXPERIMENTS: tuple[Experiment, ...] = (EXP011,)
