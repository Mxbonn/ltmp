import argparse
from pathlib import Path

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Reduce checkpoints to only contain threshold parameters,"
    "as all other parameters are unchanged from the original checkpoint."
)
parser.add_argument("dir", type=str, help="path to the directory containing checkpoints")


def main():
    args = parser.parse_args()

    checkpoint_paths = Path(args.dir).glob("**/*.pth.tar")

    for p in tqdm(checkpoint_paths):
        checkpoint = torch.load(p)
        for key in list(checkpoint["state_dict"].keys()):
            if "threshold" not in key:
                checkpoint["state_dict"].pop(key)
        torch.save(checkpoint, p)


if __name__ == "__main__":
    main()
