"""Count dense-network parameters without Torch, CUDA or a solver session."""

import config
from network_analysis import save_analysis

if __name__ == "__main__":
    save_analysis(config.ROOT / "docs")
