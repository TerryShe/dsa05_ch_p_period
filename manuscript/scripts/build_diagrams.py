"""Draw publication schematics of the implemented scenario and recurrent MAPPO.

These vector diagrams describe the model; positions and phases are illustrative,
not sampled experiment results. No policies or experiments are executed.
"""

from math import hypot
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#263747"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#00846B"
GRAY = "#687887"
LIGHT = "#EAF3FA"


def label(ax, x, y, text, size=8, **kwargs):
    return ax.text(x, y, text, fontsize=size, ha="center", va="center",
                   color=INK, linespacing=1.2, **kwargs)


def box(ax, x, y, w, h, text, face=LIGHT, edge=BLUE, size=8, **kwargs):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=.8, facecolor=face, edgecolor=edge, **kwargs,
    ))
    label(ax, x+w/2, y+h/2, text, size=size)


def arrow(ax, start, end, color=INK, dashed=False, curve=0, lw=.9):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        connectionstyle=f"arc3,rad={curve}", linewidth=lw, color=color,
        linestyle=(0, (3, 2)) if dashed else "-", shrinkA=1.5, shrinkB=1.5,
    ))


def wire(ax, points, color=INK, dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=.8,
            linestyle=(0, (3, 2)) if dashed else "-", solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color, dashed=dashed)


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def scenario():
    fig, ax = canvas(3.45, 3.02)
    label(ax, 1.725, 2.91, "(a) Shared uplink and delayed AP feedback", 8.5,
          fontweight="bold")
    ax.add_patch(Circle((1.02, 2.04), .77, facecolor="#F5F8FA",
                        edgecolor="#9CAAB5", linewidth=.8, linestyle=(0, (3, 2))))
    ap = (1.02, 2.04)
    ues = [(.39, 2.54, "1", BLUE), (1.62, 2.55, "2", BLUE),
           (.38, 1.53, "3", ORANGE), (1.66, 1.51, "4", GRAY)]
    for x, y, name, color in ues:
        if name != "4":
            dx, dy = x-ap[0], y-ap[1]
            norm = hypot(dx, dy)
            arrow(ax, (x, y), (ap[0]+.24*dx/norm, ap[1]+.24*dy/norm),
                  color=color, lw=1.5)
        ax.add_patch(FancyBboxPatch((x-.10, y-.13), .20, .26,
                                  boxstyle="round,pad=0.01,rounding_size=0.025",
                                  edgecolor=color, facecolor="white", linewidth=1))
        ax.plot([x-.045, x+.045], [y-.08, y-.08], color=color, lw=.7)
        label(ax, x, y+.005, name, 8)
        label(ax, x, y+.23 if y > 2 else y-.23, f"UE {name}", 7)
    # A common receiver, with antenna and radio arcs.
    ax.add_patch(Rectangle((ap[0]-.17, ap[1]-.13), .34, .26,
                           facecolor="white", edgecolor=INK, lw=1))
    label(ax, *ap, "AP", 9, fontweight="bold")
    ax.plot([ap[0], ap[0]], [ap[1]+.13, ap[1]+.31], color=INK, lw=1)
    for r in (.13, .21):
        ax.add_patch(FancyArrowPatch((ap[0]-r, ap[1]+.29),
                                    (ap[0]+r, ap[1]+.29),
                                    arrowstyle="-", connectionstyle="arc3,rad=-.6",
                                    color=INK, lw=.7))
    # Dashed arrow is feedback received even by a silent UE.
    arrow(ax, (1.2, 2.04), (1.57, 1.64), color=GREEN, dashed=True, curve=-.35)
    label(ax, .66, 2.29, "$k=1$", 7, bbox=dict(facecolor="white", edgecolor="none", pad=.1))
    label(ax, 1.43, 2.28, "$k=1$", 7, bbox=dict(facecolor="white", edgecolor="none", pad=.1))
    label(ax, .62, 1.8, "$k=2$", 7, bbox=dict(facecolor="white", edgecolor="none", pad=.1))
    label(ax, 1.88, 1.5, "OFF", 7)
    label(ax, 2.67, 2.63, "Each UE chooses", 8, fontweight="bold")
    label(ax, 2.67, 2.35, "subchannel + power\nor silence (OFF)", 8)
    box(ax, 2.08, 1.53, 1.2, .56,
        "AP feedback\nprevious RSSI $z_k$\nprivate service $b_i$",
        face="#EAF6F1", edge=GREEN, size=7.4)
    label(ax, 2.68, 1.31, "Same channel: interference", 6.8)

    ax.plot([.08, 3.37], [1.13, 1.13], color="#C7D0D7", lw=.6)
    label(ax, 1.725, .99, "(b) Periodic packets with offset arrivals", 8.5,
          fontweight="bold")
    left, right = .62, 3.28
    def slot(t):
        return left + t/50 * (right-left)

    for period, phase, y, color in [(10, 2, .76, BLUE), (20, 5, .51, ORANGE),
                                  (40, 8, .26, GREEN)]:
        label(ax, .29, y, f"$P={period}$", 7.4)
        ax.plot([left, right], [y, y], color="#A8B3BD", lw=.55)
        ax.add_patch(Rectangle((slot(phase), y-.062), slot(period)-left, .124,
                               facecolor=color, alpha=.16, linewidth=0))
        # Only the first packet window is shaded. Release markers repeat.
        for arrival in range(phase, 51, period):
            ax.plot(slot(arrival), y, marker="^", markersize=3.6, color=color)
        deadline = phase+period
        ax.plot([slot(deadline), slot(deadline)], [y-.09, y+.09], color=color, lw=1)
    for t in (0, 10, 20, 30, 40, 50):
        label(ax, slot(t), .065, str(t), 6.5)
    label(ax, .26, .065, "Slot", 7)
    save(fig, "system_scenario")


def framework():
    fig, ax = canvas(7.05, 2.90)
    label(ax, 3.525, 2.78,
          "(a) Distributed execution: shared actor weights, private memory at each UE", 8.5,
          fontweight="bold")
    # One branch represents every k; encoder, GRU and channel head share weights.
    box(ax, .05, 1.96, .88, .56, "Local features\n$x_{ik}$: 13, $u_i$: 13", size=7.3)
    box(ax, 1.12, 1.96, .77, .56, "Encoder $f$\n26 $\\to$ 64\ntanh", size=7.3)
    box(ax, 2.10, 1.96, .82, .56, "Shared GRU\n64 units\n$h_{ik}(t)$", size=7.3)
    box(ax, 3.15, 1.96, 1.12, .56,
        "$\\phi$ $\\to$ mean over $K$\nconcat $u_i$ $\\to$ $g$\n$v_i(t)$: 64", size=7.1)
    box(ax, 4.56, 1.99, 1.03, .53,
        "Channel head $\\psi$\n$[h_{ik},v_i,u_i]$\n5 logits / channel", size=7.1)
    box(ax, 4.56, 1.25, 1.03, .48,
        "OFF head $\\omega$\n$[v_i,u_i]$ $\\to$ 1 logit", size=7.3)
    box(ax, 5.95, 1.56, 1.02, .90,
        "Concat: 21 logits\nLegal-action mask\nSoftmax + sample\nOFF or $(k_i,p_i)$",
        face="#EAF6F1", edge=GREEN, size=7.5)
    for x1, x2 in ((.93, 1.12), (1.89, 2.10), (2.92, 3.15), (4.27, 4.56)):
        arrow(ax, (x1, 2.23), (x2, 2.23))
    wire(ax, [(4.39, 2.23), (4.39, 1.49), (4.56, 1.49)])
    wire(ax, [(5.59, 2.24), (5.76, 2.24), (5.95, 2.24)])
    wire(ax, [(5.59, 1.49), (5.77, 1.49), (5.77, 1.77), (5.95, 1.77)])
    # Local h_ik reaches the power head in addition to the pooled summary.
    wire(ax, [(2.79, 2.52), (2.79, 2.63), (5.07, 2.63), (5.07, 2.52)])
    label(ax, 3.86, 2.64, "$h_{ik}$", 7,
          bbox=dict(facecolor="white", edgecolor="none", pad=1))
    # Explicit private recurrent state from one slot to the next.
    box(ax, 2.09, 1.31, .84, .32, "$h_{ik}(t-1)$", face="white", edge=GRAY, size=7.4)
    arrow(ax, (2.51, 1.63), (2.51, 1.96), color=GRAY)
    wire(ax, [(2.80, 1.96), (2.80, 1.79), (3.07, 1.79),
              (3.07, 1.47), (2.93, 1.47)], color=GRAY)
    label(ax, 3.28, 1.48, "$z^{-1}$", 7)
    # The raw UE context is supplied to the encoder, pooling map and both heads.
    wire(ax, [(.14, 1.96), (.14, 1.08), (3.71, 1.08), (3.71, 1.96)], color=BLUE)
    wire(ax, [(3.71, 1.08), (5.08, 1.08), (5.08, 1.25)], color=BLUE)
    label(ax, 1.00, 1.09, "$u_i$", 7.5,
          bbox=dict(facecolor="white", edgecolor="none", pad=1))
    label(ax, .71, 1.57, "One stream per\nsubchannel $k$", 7)
    label(ax, 3.525, .90,
          "GRU updates during idle slots; empty queues are masked to OFF.  $K=4$, $|\\mathcal{P}|=5$.", 7.2)
    ax.plot([.04, 7.01], [.76, .76], color=GRAY, lw=.7, linestyle=(0, (4, 3)))
    # Training data do not enter the execution actor as global observations.
    label(ax, 3.525, .64, "(b) Centralized training only", 8.2, fontweight="bold")
    box(ax, .05, .09, 1.72, .37, "All UE features, true gains,\npenalty weights", face="#F1F3F5", edge=GRAY, size=7.3)
    box(ax, 1.98, .09, 1.42, .37, "Critic: 256 $\\to$ 256 $\\to$ $V$\ntanh MLP", face="#F1F3F5", edge=GRAY, size=7.3)
    box(ax, 3.61, .09, 1.43, .37, "Team reward + rollouts\n$V$ $\\to$ GAE advantages", face="#F1F3F5", edge=GRAY, size=7.3)
    box(ax, 5.25, .09, 1.72, .37, "Clipped PPO actor update\nand critic value loss", face="#F1F3F5", edge=GRAY, size=7.3)
    for x1, x2 in ((1.77, 1.98), (3.4, 3.61), (5.04, 5.25)):
        arrow(ax, (x1, .275), (x2, .275), color=GRAY)
    save(fig, "mappo_framework")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    scenario()
    framework()
    print("Generated scenario and network diagrams (PDF, SVG, PNG).")


if __name__ == "__main__":
    main()
