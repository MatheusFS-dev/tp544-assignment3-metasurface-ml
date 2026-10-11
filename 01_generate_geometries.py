"""Save all candidate masks and codes offline, with a per-layout checkpoint."""

import config
from geometry_manifest import generate_geometries

if __name__ == "__main__":
    generate_geometries(config.GEOMETRY_DIR, config.PHYSICS["pixel_m"])
