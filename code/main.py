import argparse

from logic_bot.logic_bot_runner import LogicBotRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-difficulty")
    parser.add_argument("-games")

    args = parser.parse_args()
    log_level, difficulty, games = (
        args.log,
        args.difficulty,
        args.games,
    )

    runner = LogicBotRunner(log_level, difficulty, games)
    moves, results, win_rate, average_turns = runner.run()

    print(f"\nTotal games played: {len(moves)}")
    print(f"Difficulty: {difficulty}")
    print(f"Win rate: {100 * win_rate:.2f}%")
    print(f"Average turns per game: {average_turns}\n")
