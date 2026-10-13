"""EXP-011 analysis adapter: the pre-registered comparisons of ``analysis.json``, as written.

Nothing is recomputed here. Each family lists rows of one paired difference
(``scripts/analyze_exp011.py``), with its label spelling out the subtraction.
"""

CORE = ("8", "10", "12")


def _row(key: str, label: str, entry: dict) -> dict:
    return {
        "key": key,
        "label": label,
        "difference": entry["difference"],
        "ratio": entry.get("ratio"),
        "ci95": entry["ci95"],
        "deployments": entry.get("deployments"),
    }


def _over_n(by_n: dict, pick=lambda entry: entry) -> list[dict]:
    return [_row(n, f"N = {n}", pick(entry)) for n, entry in by_n.items()]


def _theta_label(theta: str) -> str:
    return "every flag read" if theta == "all" else f"θ = {theta.replace('-', '−')} dB"


def comparisons(analysis: dict) -> list[dict]:
    families = [
        {
            "id": "D_N",
            "title": "U − M (main rule D_N)",
            "note": f"decision on N = 8, 10, 12: {analysis['decision']}",
            "scale": "rate",
            "core": list(CORE),
            "rows": _over_n(analysis["D_N"]),
        },
        {
            "id": "E_N",
            "title": "U − (M+Y) (coordination E_N)",
            "note": "enters the final confirmation: "
            + analysis["coordination"]["enters_final_confirmation"],
            "scale": "rate",
            "rows": _over_n(analysis["coordination"]["E_N"]),
        },
    ]
    for other, by_n in analysis["content"].items():
        families.append(
            {
                "id": f"content-{other}",
                "title": f"U − {other} (content)",
                "scale": "rate",
                "rows": _over_n(by_n),
            }
        )
    for hide in ("all", "nonurgent", "urgent"):
        families.append(
            {
                "id": f"mechanism-{hide}",
                "title": f"U-hide-{hide} − U (mechanism)",
                "scale": "rate",
                "rows": _over_n(analysis["mechanism"], lambda entry, h=hide: entry[h]),
            }
        )
    for method in ("U", "M+Y"):
        for n, entry in analysis["receive"][method].items():
            families.append(
                {
                    "id": f"receive-{method}-{n}",
                    "title": f"{method}@θ − M at N = {n} (receive condition)",
                    "note": f"boundary: {entry['boundary'] or 'none'}",
                    "scale": "rate",
                    "rows": [
                        _row(theta, _theta_label(theta), row)
                        for theta, row in entry["by_theta"].items()
                    ],
                }
            )
    for key, title in (("U0_minus_M", "U0 − M"), ("U0_minus_U_at_0dB", "U0 − U@0")):
        families.append(
            {
                "id": key,
                "title": f"{title} (receive condition)",
                "scale": "rate",
                "rows": _over_n(analysis["receive"]["U0"], lambda entry, k=key: entry[k]),
            }
        )
    for key, title, scale in (
        ("all_ue_mean_rate_U_minus_M", "U − M, all-UE mean rate (guard)", "rate"),
        ("log_w_U_minus_M", "U − M, log10(w + δ) (sensitivity)", "log"),
        ("excluding_infeasible_for_all", "U − M, excluding infeasible-for-all", "rate"),
    ):
        families.append(
            {
                "id": f"sensitivity-{key}",
                "title": title,
                "scale": scale,
                "rows": _over_n(analysis["sensitivity"], lambda entry, k=key: entry[k]),
            }
        )
    return families


def tables(analysis: dict) -> dict:
    """Per-method tables of the analysis (point estimates; no per-method intervals exist)."""
    return {
        "P2": analysis["P2"],
        "P1": analysis["P1"],
        "p90_w": analysis["p90_w"],
        "all_ue_mean_rate": analysis["all_ue_mean_rate"],
        "N_star_on_grid": analysis["N_star_on_grid"],
    }
