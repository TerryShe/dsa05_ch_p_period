"""Fixed-rollout training diagnostics of EXP-DSA05-008 (design §8). Descriptive only.

For each checkpoint, a trainer with the checkpoint's networks, update options
and ValueNorm statistics, but R0's mix of numbers of UEs (``--own-env-ues``: the
checkpoint's own environments, e.g. 14 x N for an EXP-009 per-N run), collects ``--rollouts``
rollouts of 128 slots from the training pool with a fixed seed (the same
episodes and fading for every checkpoint; the actions differ). Nothing is
trained and the ValueNorm statistics do not move. Per number of UEs N, over the
UE-slot samples with a packet (the samples of the actor loss):

* Advantage variance decomposition. With delta_t = r_t + gamma V_{t+1} - V_t,
  the GAE splits into a reward part A^r_t = sum_l (gamma lam)^l r_{t+l} and a
  value part A^v_t = sum_l (gamma lam)^l (gamma V_{t+l+1} - V_{t+l}), with the
  truncation of the training GAE; A = A^r + A^v. Reported: Var(A), the shares
  of Var(A^r), Var(A^v) and 2 Cov(A^r, A^v), their correlation, and the number
  of slots with a nonzero reward. Reward units, before normalization.
* Actor-gradient diagnostics. The policy-gradient surrogate at ratio 1 with the
  checkpoint's advantage normalization and loss weighting (no entropy term) is
  split into per-N contributions that sum to the total loss; per N: the norm of
  the mean contribution gradient over the rollouts, the mean per-rollout norm,
  its cosine with the total gradient and its projection share on it (shares sum
  to 1 and can be negative; they are not additive "contributions" of norms).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from dsa_period.config import load_scenario
from dsa_period.marl.networks import masked_log_probs
from dsa_period.marl.ppo import PPOSettings
from dsa_period.marl.trainer_v2 import (
    CapacityConstraint,
    CapacityTrainer,
    CapacityTrainingSettings,
    UpdateSettings,
)

CONFIG = Path("configs/scenario-v2.toml")
R0_ENV_UES = (4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 16, 16, 20, 20)
DIAGNOSTIC_SEED = 8_008_001


def decompose(rewards, values, next_values, truncated, gamma: float, lam: float):
    """Reward and value parts of the GAE of ``ppo.discounted_gae`` (same truncation)."""
    reward_part = torch.empty_like(rewards)
    value_part = torch.empty_like(rewards)
    trace_r = torch.zeros_like(rewards[0])
    trace_v = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        keep = gamma * lam * (~truncated[t]).to(rewards.dtype)
        trace_r = rewards[t] + keep * trace_r
        trace_v = gamma * next_values[t] - values[t] + keep * trace_v
        reward_part[t], value_part[t] = trace_r, trace_v
    return reward_part, value_part


def build_trainer(
    checkpoint: Path, seed: int = DIAGNOSTIC_SEED, own_env_ues: bool = False
) -> CapacityTrainer:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
    env_ues = tuple(state["settings"]["env_ues"]) if own_env_ues else R0_ENV_UES
    settings = CapacityTrainingSettings(**{**state["settings"], "env_ues": env_ues})
    trainer = CapacityTrainer(
        load_scenario(CONFIG, max(env_ues)),
        settings,
        PPOSettings(**manifest["resolved"]["ppo"]),
        CapacityConstraint(**state["constraint"]),
        seed,
        UpdateSettings(**state.get("update", {})),
    )
    trainer.ppo.load_state_dict(state["ppo"], optimizers=False)
    return trainer


def surrogate_parts(trainer: CapacityTrainer, rollout, per_ue, active) -> dict[int, torch.Tensor]:
    """Per-N parts of the ratio-1 policy-gradient loss; they sum to the full loss."""
    ppo = trainer.ppo
    logits, _ = ppo.actor.sequence(
        torch.from_numpy(rollout.tokens),
        torch.from_numpy(rollout.context),
        rollout.initial_hidden,
        torch.from_numpy(rollout.starts)[:, ppo.env_of_ue],
    )
    log_probs = masked_log_probs(logits, torch.from_numpy(rollout.legal))
    chosen = log_probs.gather(-1, torch.from_numpy(rollout.actions).unsqueeze(-1)).squeeze(-1)
    terms = -(per_ue.detach() * chosen)  # [T, U]
    if ppo.options.actor_loss_weighting == "env":
        counts = torch.zeros(ppo.num_envs).index_add(
            0, ppo.env_of_ue, active.sum(0).to(torch.float32)
        )
        valid = int((counts > 0).sum())
        weights = (1.0 / counts.clamp(min=1.0))[ppo.env_of_ue] / valid
    else:
        weights = torch.full((ppo.env_of_ue.numel(),), 1.0 / int(active.sum()))
    weighted = (terms * weights) * active
    return {n: weighted[:, ppo.n_of_ue == n].sum() for n in ppo.groups}


def diagnose(job: tuple) -> tuple[str, dict]:
    checkpoint, rollouts, own_env_ues = job
    torch.set_num_threads(1)
    trainer = build_trainer(Path(checkpoint), own_env_ues=own_env_ues)
    ppo, s = trainer.ppo, trainer.ppo.settings
    params = [p for p in ppo.actor.parameters()]
    moments = {n: np.zeros(7) for n in ppo.groups}  # n, sum r, sum v, rr, vv, rv, nonzero slots
    grad_sum = {n: None for n in ppo.groups}
    grad_norms = {n: 0.0 for n in ppo.groups}
    total_sum = None
    max_error = 0.0
    for _ in range(rollouts):
        rollout, _ = trainer.collect()
        prepared = ppo.advantages(rollout, update_stats=False)
        reward_part, value_part = decompose(
            torch.from_numpy(rollout.rewards),
            prepared["values"],
            prepared["next_values"],
            torch.from_numpy(rollout.truncated),
            s.gamma,
            s.gae_lambda,
        )
        max_error = max(
            max_error, float((reward_part + value_part - prepared["advantages"]).abs().max())
        )
        active = prepared["active"]
        for n in ppo.groups:
            cols = ppo.n_of_ue == n
            act = active[:, cols]
            r = reward_part[:, ppo.env_of_ue][:, cols][act].double().numpy()
            v = value_part[:, ppo.env_of_ue][:, cols][act].double().numpy()
            envs = (ppo.n_of_env == n).numpy()
            nonzero = int((rollout.rewards[:, envs] != 0).sum())
            moments[n] += [r.size, r.sum(), v.sum(), r @ r, v @ v, r @ v, nonzero]
        parts = surrogate_parts(trainer, rollout, prepared["per_ue"], active)
        for i, n in enumerate(ppo.groups):
            grads = torch.autograd.grad(
                parts[n], params, retain_graph=i < len(ppo.groups) - 1, allow_unused=True
            )
            vector = torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, params)
                ]
            ).double()
            grad_sum[n] = vector if grad_sum[n] is None else grad_sum[n] + vector
            grad_norms[n] += float(vector.norm())
            total_sum = vector.clone() if total_sum is None else total_sum + vector
    total = total_sum / rollouts
    out = {"rollouts": rollouts, "gae_split_max_abs_error": max_error, "by_n": {}}
    for n in ppo.groups:
        count, sr, sv, rr, vv, rv, nonzero = moments[n]
        row: dict = {"samples": int(count), "nonzero_reward_slots": int(nonzero)}
        if count > 0:
            mr, mv = sr / count, sv / count
            var_r, var_v = rr / count - mr**2, vv / count - mv**2
            cov = rv / count - mr * mv
            var_a = var_r + var_v + 2 * cov
            row.update(
                var_advantage=var_a,
                share_reward=var_r / var_a if var_a > 0 else None,
                share_value=var_v / var_a if var_a > 0 else None,
                share_cross=2 * cov / var_a if var_a > 0 else None,
                corr_reward_value=cov / np.sqrt(var_r * var_v) if var_r > 0 and var_v > 0 else None,
            )
        mean = grad_sum[n] / rollouts
        row.update(
            grad_mean_norm=float(mean.norm()),
            grad_rollout_norm_mean=grad_norms[n] / rollouts,
            grad_cosine_with_total=float(mean @ total / (mean.norm() * total.norm()))
            if mean.norm() > 0
            else None,
            grad_projection_share=float(mean @ total / (total @ total)),
        )
        out["by_n"][str(n)] = row
    out["grad_total_norm"] = float(total.norm())
    return checkpoint, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--own-env-ues", action="store_true", help="the checkpoint's environments")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    jobs = [(str(c), args.rollouts, args.own_env_ues) for c in args.checkpoints]
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(diagnose, jobs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": DIAGNOSTIC_SEED, "results": results}, indent=2))
    for checkpoint, out in results.items():
        print(checkpoint, f"(GAE split error {out['gae_split_max_abs_error']:.1e})")
        for n, row in out["by_n"].items():
            share = row.get("share_value")
            print(
                f"  N={n:>2} samples {row['samples']:>7} value share "
                f"{'-' if share is None else f'{share:.2f}'} grad cos "
                f"{row['grad_cosine_with_total'] or 0:+.2f} proj {row['grad_projection_share']:+.3f}"
            )


if __name__ == "__main__":
    main()
