"""Measured benchmark; writes full held-out results instead of an AGI verdict."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noema.self_improvement.cli import main

if __name__ == "__main__":
    main()
