from pathlib import Path
from vibeweights.language_weights import build
build(Path(__file__).resolve().parents[1])
