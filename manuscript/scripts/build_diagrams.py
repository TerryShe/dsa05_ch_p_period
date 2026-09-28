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
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
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
    fig, ax = canvas(3.45, 2.72)
    label(ax, 1.725, 2.62, "Shared actor; private memory at each UE", 8.2,
          fontweight="bold")
    box(ax, .05, 2.08, .91, .34, "Local input\n$[x_{ik},u_i]$: 26", size=7)
    box(ax, 1.13, 2.08, .96, .34, "Encoder + GRU\n$h_{ik}$: 64", size=7)
    box(ax, 2.31, 2.08, 1.09, .34, "$\\phi$; mean over $K$\n$g(\\cdot,u_i)$: $v_i$", size=7)
    arrow(ax, (.96, 2.25), (1.13, 2.25))
    arrow(ax, (2.09, 2.25), (2.31, 2.25))
    # Recurrent input comes from the previous slot of this UE/subchannel.
    label(ax, 1.60, 1.88, "$h_{ik}(t-1)$", 7)
    arrow(ax, (1.60, 1.96), (1.60, 2.08), color=GRAY)
    box(ax, .05, 1.17, 1.63, .46,
        "Shared power head $\\psi$\n$[h_{ik},v_i,u_i]$ $\\to$ 5 logits / $k$", size=7)
    box(ax, 1.90, 1.17, 1.50, .46,
        "OFF head $\\omega$\n$[v_i,u_i]$ $\\to$ 1 logit", size=7)
    wire(ax, [(2.85, 2.08), (2.85, 1.76), (.86, 1.76), (.86, 1.63)])
    arrow(ax, (2.65, 1.76), (2.65, 1.63))
    box(ax, .38, .60, 2.69, .32,
        "21 logits: mask $\\to$ softmax $\\to$ sample", face="#EAF6F1", edge=GREEN, size=7.4)
    wire(ax, [(.86, 1.17), (.86, 1.04), (1.32, 1.04), (1.32, .92)])
    wire(ax, [(2.65, 1.17), (2.65, 1.04), (2.13, 1.04), (2.13, .92)])
    ax.plot([.04, 3.41], [.47, .47], color=GRAY, lw=.7, linestyle=(0, (4, 3)))
    label(ax, 1.725, .33, "Training only: global features + true gains", 7.2)
    label(ax, 1.725, .12, "Critic $\\to$ GAE + team reward $\\to$ clipped PPO", 7.2)
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
