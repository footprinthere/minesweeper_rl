import argparse
import os

from pprint import pformat

from game import MineSweeperEnv
from model import ModelParameter
from trainer import MineSweeperTrainer, TrainParameter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)

    parser.add_argument("--board_height", type=int, required=True)
    parser.add_argument("--board_width", type=int, required=True)
    parser.add_argument("--n_mines", type=int, required=True)

    parser.add_argument("--n_channels", type=int, required=True)
    parser.add_argument("--model_depth", type=int, required=True)
    parser.add_argument("--ff_dim", type=int, required=True)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--q_sample_size", type=int, required=True)
    parser.add_argument("--use_action_mask", action="store_true")

    parser.add_argument("--n_episodes", type=int, required=True)

    args = parser.parse_args()

    # Save arguments
    os.makedirs(args.project_dir, exist_ok=True)
    with open(os.path.join(args.project_dir, "args.txt"), "w") as f:
        f.write(pformat(vars(args)))

    # Prepare game environment
    print("Preparing game environment...")
    env = MineSweeperEnv(
        board_height=args.board_height,
        board_width=args.board_width,
        n_mines=args.n_mines,
    )

    # Prepare trainer
    print("Preparing trainer...")
    model_param = ModelParameter(
        board_size=args.board_height * args.board_width,
        n_channels=args.n_channels,
        cnn_depth=args.model_depth,
        ff_dim=args.ff_dim,
    )
    train_param = TrainParameter(
        batch_size=args.batch_size,
        q_sample_size=args.q_sample_size,
        use_action_mask=args.use_action_mask,
    )
    trainer = MineSweeperTrainer(
        env=env,
        model_param=model_param,
        train_param=train_param,
        project_dir=args.project_dir,
    )

    # Train
    print("Start training...")
    if args.log_file is None:
        output_file = None
    elif args.log_file == "stdin":
        output_file = "stdin"
    else:
        output_file = os.path.join(args.project_dir, args.log_file)
    trainer.train(n_episodes=args.n_episodes, output_file=output_file)

    # Save model states
    print("Training completed. Saving model states...")
    trainer.save_models()

    # Plot results
    print("Plotting results...")
    trainer.plot_logs()


if __name__ == "__main__":
    main()
