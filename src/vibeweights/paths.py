"""Portable checkout paths; no dependency on current working directory."""
from pathlib import Path
import os
from .abi import ContractError

def project_root():
    root = Path(os.environ.get('VIBEWEIGHTS_ROOT', Path(__file__).resolve().parents[2])).resolve()
    if not (root/'catalog'/'manifests').is_dir():
        raise ContractError('Set VIBEWEIGHTS_ROOT to the unpacked project root')
    return root

def data_root(root):
    path = Path(root).resolve()/'Data'
    path.mkdir(parents=True, exist_ok=True)
    return path
