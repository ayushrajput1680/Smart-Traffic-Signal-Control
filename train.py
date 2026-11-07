import argparse
import os
import numpy as np
import pandas as pd
import torch

from agent import DQNAgent
from env import MultiIntersectionEnv


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    env = MultiIntersectionEnv(num_intersections=4, episode_length=args.episode_length)
    agent_ids = env.agent_ids

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_agent():
        return DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
        )

    if args.algo == "shared_dqn":
        shared_agent = make_agent()
        agents = {aid: shared_agent for aid in agent_ids}
    elif args.algo == "independent_dqn":
        agents = {aid: make_agent() for aid in agent_ids}
    else:
        raise ValueError(f"Unknown algorithm mode: {args.algo}")

    metrics_log = []
    epsilon, epsilon_min = 1.0, 0.05
    epsilon_decay = (1.0 - epsilon_min) / (args.episodes * 0.8)

    for episode in range(1, args.episodes + 1):
        states = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        losses = []

        while True:
            masks = env.get_action_masks()
            actions = {
                aid: agents[aid].select_action(states[aid], epsilon=epsilon, valid_mask=masks[aid])
                for aid in agent_ids
            }
            next_states, rewards, dones, info = env.step(actions)

            for aid in agent_ids:
                agents[aid].replay_buffer.push(
                    states[aid], actions[aid], rewards[aid], next_states[aid], float(dones[aid])
                )
                ep_reward += rewards[aid]
                if len(agents[aid].replay_buffer) >= args.batch_size:
                    losses.append(agents[aid].update())

            states = next_states
            if any(dones.values()):
                break

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        metrics_log.append({
            "episode": episode,
            "reward": ep_reward,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "avg_waiting_time": info["avg_waiting_time"],
            "avg_queue_length": info["avg_queue_length"],
        })
        print(f"Ep [{episode}/{args.episodes}] | Return: {ep_reward:.1f} | "
              f"Wait: {info['avg_waiting_time']:.2f}s | Queue: {info['avg_queue_length']:.2f}")

    csv_path = os.path.join(args.log_dir, f"{args.algo}_seed{args.seed}.csv")
    pd.DataFrame(metrics_log).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["independent_dqn", "shared_dqn"], default="shared_dqn")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode_length", type=int, default=3600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="results")
    train(parser.parse_args())
