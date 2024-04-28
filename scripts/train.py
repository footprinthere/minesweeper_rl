import argparse
import os

from pprint import pformat

from game import MineSweeperEnv
from trainer import MineSweeperTrainer, MaxQValTracker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str)

    parser.add_argument("--board_height", type=int)
    parser.add_argument("--board_width", type=int)
    parser.add_argument("--n_mines", type=int)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--q_sample_size", type=int)
    parser.add_argument("--use_action_mask", action="store_true")

    parser.add_argument("--n_channels", type=int)
    parser.add_argument("--model_depth", type=int)

    args = parser.parse_args()

    # Save arguments
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        f.write(pformat(vars(args)))

    # Prepare game environment
    print("Preparing game environment...")
    env = MineSweeperEnv(
        board_height=args.board_height,
        board_width=args.board_width,
        n_mines=args.n_mines,
    )

    # Prepare trainer
    print("Preparing Q tracker...")
    q_tracker = MaxQValTracker(env=env)
    q_tracker.collect_samples(size=args.q_sample_size)

    print("Preparing trainer...")
    trainer = MineSweeperTrainer(
        env=env,
        board_size=args.board_height * args.board_width,
        n_channels=args.n_channels,
        model_depth=args.model_depth,
        batch_size=args.batch_size,
        q_tracker=q_tracker,
        use_action_mask=args.use_action_mask,
    )

    # Train
    print("Start training...")
    trainer.train(n_episodes=args.n_episodes)

    # Plot results
    print("Training completed. Plotting results...")
    trainer.plot_logs(log_dir=args.log_dir)


if __name__ == "__main__":
    main()
