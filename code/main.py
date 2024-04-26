import argparse

from tasks.task_1.task_1 import Task1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-file")
    parser.add_argument("-games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    log_level, file, games, width, height, mines = (
        args.log,
        args.file,
        args.games,
        args.width,
        args.height,
        args.mines,
    )

    print(
        f"Running Task1 with: log_level={log_level}, file={file}, games={games}, width={width}, height={height}, mines={mines}"
    )

    task = Task1(log_level, file, games, width, height, mines)
    task.run()

    print("Done\n")
