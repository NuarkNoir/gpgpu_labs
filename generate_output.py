from random import uniform
from typing import Tuple, List
from argparse import ArgumentParser, Namespace
from pathlib import PurePath

def generate_output(size: int, lb: float, ub: float) -> Tuple[List[float], List[float], List[float]]:
    a = [uniform(lb, ub) for _ in range(size)]
    b = [uniform(lb, ub) for _ in range(size)]
    s = [a[i] + b[i] for i in range(size)]

    return a, b, s


def main(args: Namespace) -> None:
    a, b, s = generate_output(args.size, args.lower_bound, args.upper_bound)

    with open(PurePath(args.output) / "data.txt", "w") as f:
        f.write(f"{args.size}\n")
        for i in range(args.size):
            f.write(f"{a[i]} {b[i]} {s[i]}\n")


if __name__ == "__main__":
    # use argparse to get params
    parser = ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=1000)
    parser.add_argument("-l", "--lower-bound", type=float, default=0.0)
    parser.add_argument("-u", "--upper-bound", type=float, default=1.0)
    parser.add_argument("-o", "--output", type=str, default=".")
    args = parser.parse_args()

    main(args)