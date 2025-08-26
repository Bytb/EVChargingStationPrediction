# EV_Research/RNN/run.py
from pathlib import Path
import argparse
from EV_Research.RNN.training.train import train_and_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True, help="Path to YAML config")
    args = ap.parse_args()
    train_and_eval(args.cfg)
