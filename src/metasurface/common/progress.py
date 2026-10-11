"""Small helpers for reporting committed stage progress."""

import time


def format_duration(seconds):
    """Format a nonnegative duration for concise console progress.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Human-readable duration.
    """
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class Progress:
    """Report progress after durable commits within one invocation."""

    def __init__(self, label, total, saved=0):
        """Start a stage timer and report its saved starting point.

        Args:
            label (str): Stage name.
            total (int): Total number of items.
            saved (int): Items already committed before this invocation.
        """
        self.label = label
        self.total = total
        self.saved = saved
        self.started = time.monotonic()
        print(f"{label}: starting with {saved}/{total} saved, ETA unknown", flush=True)

    def committed(self, count, detail=""):
        """Report a successfully committed count and estimated remaining time.

        Args:
            count (int): Current durable item count.
            detail (str): Optional item identifier.
        """
        elapsed = time.monotonic() - self.started
        advanced = count - self.saved
        eta = format_duration(elapsed / advanced * (self.total - count)) if advanced > 0 else "unknown"
        suffix = f" ({detail})" if detail else ""
        print(f"{self.label}: {count}/{self.total} saved{suffix}, elapsed {format_duration(elapsed)}, ETA {eta}", flush=True)
