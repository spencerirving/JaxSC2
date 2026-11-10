# Proximal Policy Optimization (PPO)

A standard, high-throughput PPO implementation for Sc2Jax.

## 🗝 Highlights
- **Architecture**: Feed-forward Categorical Actor-Critic.
- **Optimization**: Multi-epoch mini-batch updates with CLIP loss.
- **Normalization**: Online RunningMeanStd for both observations and value returns.

## ⚙️ Key Hyperparameters
- `LEARN_RATE`: Default `3e-4`.
- `ENTROPY_COEFF`: Default `0.01` (crucial for exploration in combat).
- `CLIP_EPS`: Default `0.2`.
- `NUM_MINIBATCHES`: Controls the gradient update granularity.

## 🛠 Extension Guide
To modify the network architecture, edit `algorithms/ppo/model.py`. The `ActorCritic` class uses Flax Linen; ensure any new layers are JAX-compatible.

## Exporting experiment results

After training with `examples/run_ppo.py`, export curves, final evaluation,
and three looping policy GIFs for each run:

```bash
python examples/ppo_results.py runs/ppo_V* --episodes 64 --gifs 3
```

Each run gets `results/<final_checkpoint>_seed_12345_n_64/` containing
`reward_curves.png`, `scalars.csv`, `evaluation.json`, and three GIFs.
Reports include mean reward, population reward standard deviation, win rates,
and individual episode results. Evaluation uses deterministic actions and saved
environment settings, including episode horizon, reward profile, and weights.
Older configs without a reward profile use legacy rewards.

GIFs reuse the scripted gallery renderer: 480×480 full-map framing, labels,
a held final outcome, and infinite looping. Checkpoints and training logs are
preserved; repeating an export replaces generated outputs at the same path.
Only load trusted pickle checkpoints. New simulator code can change evaluation
behavior for old checkpoints even when saved settings match.

Export curves during training with `--curves-only`, or select an intermediate
checkpoint with `--checkpoint PATH` and a single run directory. Set `--gifs 0`
to skip rendering. View live curves using `tensorboard --logdir runs`.

The glob above includes older runs too. To process only updated experiments,
pass their exact run directories or use a date-specific glob such as
`runs/ppo_V*_20261109T*`.
