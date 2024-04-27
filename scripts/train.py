import argparse

from game import MineSweeperEnv
from train import MineSweeperTrainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str)

    parser.add_argument("--board_height", type=int)
    parser.add_argument("--board_width", type=int)
    parser.add_argument("--n_mines", type=int)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_episodes", type=int)

    parser.add_argument("--n_channels", type=int)
    parser.add_argument("--model_depth", type=int)

    args = parser.parse_args()

    # Prepare game environment
    print("Preparing game environment...")
    env = MineSweeperEnv(
        board_height=args.board_height,
        board_width=args.board_width,
        n_mines=args.n_mines,
    )

    # Prepare trainer
    print("Preparing trainer...")
    trainer = MineSweeperTrainer(
        env=env,
        board_size=args.board_height * args.board_width,
        n_channels=args.n_channels,
        model_depth=args.model_depth,
        batch_size=args.batch_size,
    )

    # Train
    print("Start training...")
    trainer.train(n_episodes=args.n_episodes)

    # Plot results
    print("Training completed. Plotting results...")
    trainer.plot_result(file_prefix=args.prefix)


if __name__ == "__main__":
    main()