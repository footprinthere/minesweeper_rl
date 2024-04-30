import sys

from game.simulator import MineSweeperSimulator


def main():
    height = int(sys.argv[1])
    width = int(sys.argv[2])
    n_mines = int(sys.argv[3])
    seed = None
    if len(sys.argv) > 4:
        seed = int(sys.argv[4])

    simulator = MineSweeperSimulator(height, width, n_mines, seed)
    simulator.play()


if __name__ == "__main__":
    main()
