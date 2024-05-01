import argparse
from task_1 import Task1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train_data_file")
    parser.add_argument("-test_data_file")
    parser.add_argument("-train_games")
    parser.add_argument("-test_games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    (
        log_level,
        train_data_file,
        test_data_file,
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
        test_data_file=test_data_file,
        train_data_file=train_data_file,
        train_games=train_games,
        test_games=test_games,
        width=width,
        height=height,
        mines=mines,
    )

    print("Train data:")
    for i, (data, label) in enumerate(
        zip(task.train_data["board_states"][:10], task.train_data["revealed_states"][:10])
    ):
        print(f"Data for game {i}:")
        for row in data:
            print(row)
        print(f"Label for board {i}")
        for row in label:
            print(row)
