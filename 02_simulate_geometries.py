"""Use lumapi to simulate previously saved layouts and checkpoint raw results."""

import config
from dataset import simulate_geometries

if __name__ == "__main__":
    simulate_geometries(config.GEOMETRY_DIR / "geometry_checkpoint.npz", config.DATA_DIR,
                        config.PHYSICS, config.BACKEND, config.GENERATION_LIMIT)
