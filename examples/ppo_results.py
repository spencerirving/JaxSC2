"""Export TwoBridge PPO curves, evaluation metrics, and looping GIFs.

Example: python examples/ppo_results.py runs/ppo_V* --episodes 64 --gifs 3
Only load trusted checkpoints: they use Python pickle.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export_curves(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run / 'logs'), size_guidance={'scalars': 0})
    events.Reload()
    tags = events.Tags().get('scalars', [])
    with (output / 'scalars.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'step', 'wall_time', 'value'])
        for tag in tags:
            for event in events.Scalars(tag):
                writer.writerow([tag, event.step, event.wall_time, event.value])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tag, title in zip(axes, ['train/episode_reward', 'eval/mean_reward'],
                              ['Training episode return', 'Evaluation episode return']):
        if tag in tags:
            points = events.Scalars(tag)
            ax.plot([p.step for p in points], [p.value for p in points])
        else:
            ax.text(.5, .5, 'No data recorded yet', ha='center', transform=ax.transAxes)
        ax.set(title=title, xlabel='Environment steps', ylabel='Reward')
        ax.grid(alpha=.3)
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(output / 'reward_curves.png', dpi=160)
    plt.close(fig)


def render_gifs(env, model, params, output, count, seed):
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from algorithms.common.utils import flatten_obs
    from JaxSC2.env.env import PerUnitAction
    from JaxSC2.scripted.evaluate import render_gif

    @jax.jit
    def act(obs, state):
        prediction = model.apply(params, flatten_obs(obs))
        heads = [jnp.argmax(prediction[name]) for name in
                 ('verb_logits', 'direction_logits', 'target_logits')]
        return PerUnitAction(state.smax_state.unit_alive[:env.num_allies],
                             *(jnp.full(env.num_allies, head, dtype=jnp.int32) for head in heads))

    step, reset = jax.jit(env.step), jax.jit(env.reset)
    for episode in range(count):
        key = jax.random.PRNGKey(seed + episode)
        key, reset_key = jax.random.split(key)
        obs, state = reset(reset_key)
        start, states = state, []
        reward_sum = 0.
        nav = combat = False
        for _ in range(env.max_steps):
            action = act(obs, state)
            key, step_key = jax.random.split(key)
            obs, state, reward, done, info = step(step_key, state, action)
            states.append(state)
            reward_sum += float(reward)
            nav |= bool(info['nav_success'])
            combat |= bool(info['combat_success'])
            if bool(done):
                break
        outcome = ('beacon_reached' if nav else 'enemies_defeated' if combat
                   else 'allies_defeated' if not np.any(state.smax_state.unit_alive[:env.num_allies])
                   else 'timeout')
        stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
        row = {'variant': env.variant_config.name, 'agent': 'PPO',
               'setting': f'seed {seed + episode}', 'steps': len(states),
               'outcome': f'{outcome} | Return {reward_sum:.2f}'}
        render_gif(env, (start, stacked), row, output / f'episode_seed_{seed + episode}.gif')
        print(f'GIF seed {seed + episode}: {outcome}, return {reward_sum:.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--gifs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--checkpoint', type=Path, help='Explicit checkpoint for a single run')
    parser.add_argument('--curves-only', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.gifs < 0:
        parser.error('--episodes must be positive and --gifs nonnegative')
    if args.checkpoint and len(args.runs) != 1:
        parser.error('--checkpoint requires exactly one run')
    for run in args.runs:
        metadata = json.loads((run / 'config.json').read_text())
        if metadata.get('algorithm') != 'PPO' or 'variant' not in metadata:
            parser.error(f'{run}: expected a TwoBridge run from examples/run_ppo.py')
        if args.curves_only:
            output = run / 'results' / 'curves'
            output.mkdir(parents=True, exist_ok=True)
            export_curves(run, output)
            print(f'Saved curves: {output}', flush=True)
            continue
        checkpoints = list((run / 'checkpoints').glob('final_*.pkl'))
        if args.checkpoint:
            checkpoint = args.checkpoint
        elif checkpoints:
            checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        else:
            parser.error(f'No final checkpoint in {run}; use --curves-only or --checkpoint PATH')
        output = run / 'results' / f'{checkpoint.stem}_seed_{args.seed}_n_{args.episodes}'
        output.mkdir(parents=True, exist_ok=True)
        export_curves(run, output)

        import jax
        import numpy as np
        from algorithms.ppo.model import MultiHeadActorCritic
        from algorithms.common.rollout import evaluate_episodes, evaluation_metrics
        from JaxSC2.env.env import JaxSC2Env

        settings = {k: metadata[k] for k in
                    ('enemy_ai', 'enemy_mode', 'enable_collision', 'enable_armor', 'enable_fow')}
        # Runs predating reward versioning used the legacy reward profile.
        settings['reward_profile'] = metadata.get('reward_profile', 'legacy')
        env = JaxSC2Env(metadata['variant'], **settings)
        env.max_steps = metadata['max_episode_steps']
        if 'reward_weights' in metadata:
            env.reward_weights = dict(metadata['reward_weights'])
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)
        model = MultiHeadActorCritic(action_dim=env.num_enemies)
        rewards, nav, combat = evaluate_episodes(env, saved['params'], model,
                                                jax.random.PRNGKey(args.seed),
                                                args.episodes, env.max_steps)
        metrics = {k: float(v) for k, v in evaluation_metrics(rewards, nav, combat).items()}
        metrics['eval/reward_std'] = float(np.std(rewards))
        report = {'variant': metadata['variant'], 'checkpoint': str(checkpoint.resolve()),
                  'step': int(saved['step']), 'seed': args.seed, 'episodes': args.episodes,
                  'environment': {**settings, 'max_episode_steps': env.max_steps,
                                  'reward_weights': env.reward_weights},
                  'metrics': metrics, 'episode_rewards': np.asarray(rewards).tolist(),
                  'nav_wins': np.asarray(nav).tolist(), 'combat_wins': np.asarray(combat).tolist()}
        (output / 'evaluation.json').write_text(json.dumps(report, indent=2))
        render_gifs(env, model, saved['params'], output, args.gifs, args.seed)
        print(f'Saved results: {output}\n{json.dumps(metrics, indent=2)}', flush=True)


if __name__ == '__main__':
    main()
