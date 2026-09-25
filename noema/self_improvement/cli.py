"""Command line entry point. Resume adds generations to a saved experiment."""
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from .runner import RunConfig, run


def main():
    parser = argparse.ArgumentParser(description="NOEMA recursive training and measured model selection")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, help="Path to checkpoint.pt; restores replay, optimizer and proposer")
    parser.add_argument("--generations", type=int, default=3, help="Generations to run now, including on resume")
    for name in ("seed", "candidates", "train_steps", "batch_size", "initial_episodes",
                 "fresh_episodes", "eval_episodes", "horizon", "patience", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu (default), cuda or cuda:N")
    args = parser.parse_args()
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = RunConfig(**saved["config"])
    else:
        config = RunConfig()
    changes = {key: value for key, value in vars(args).items()
               if key not in {"output", "resume"} and value is not None}
    config = replace(config, **changes)
    output = args.output or (args.resume.parent if args.resume else Path("runs/self_improvement"))
    try:
        run(config, output, args.resume)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
