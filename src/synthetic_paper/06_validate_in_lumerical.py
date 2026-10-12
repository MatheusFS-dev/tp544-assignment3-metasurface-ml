"""Optionally validate one exported synthetic-paper design in Lumerical."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Run the sole solver-backed stage when explicitly enabled."""
    if not config.SYNTHETIC_PAPER_RUN_VALIDATION:
        raise RuntimeError("Synthetic-paper Lumerical validation is disabled in config.py.")
    from metasurface.synthetic_paper.validation import validate_export
    with timed_part("fresh_lumerical_validation"):
        validate_export(config.SYNTHETIC_PAPER_ROOT / "design")


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "06_validate_in_lumerical", _run)
