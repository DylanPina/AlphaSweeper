import argparse

from tasks.task_1.task_1 import Task1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train_data_file")
    parser.add_argument("-test_data_file")
    parser.add_argument("-test_games")
    parser.add_argument("-train_games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    (
        log_level,
        test_data_file,
        train_data_file,
        train_games,
        test_games,
        width,
        height,
        mines,
    ) = (
        args.log,
        args.train_data_file,
        args.test_data_file,
        args.train_games,
        args.test_games,
        args.width,
        args.height,
        args.mines,
    )

    task = Task1(
        log_level,
        test_data_file,
        train_data_file,
        test_games,
        train_games,
        width,
        height,
        mines,
    )

    train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
        task.train()
    )
    task.plot(train_losses, test_losses, train_accuracies, test_accuracies)
    print(
        f"Network output: {task.get_next_move([[0, 1, -2, -2, 1, 2, -1, -2, -2], [0, 2, -2, -2, -2, -2, -2, -2, -2], [0, 1, -2, -2, -2, -2, -2, -2, 1], [0, 1, 1, 1, 1, 1, 1, 1, -2], [0, 0, 0, 0, 0, 0, 0, 2, -2], [0, 0, 0, 0, 1, 1, 1, 1, -2], [0, 1, 2, 2, 2, -2, -2, -2, 1], [0, 1, -2, -2, -2, -2, 1, 0, 0], [0, 1, -2, -2, 1, 0, 0, 0, 0]])}"
    )
