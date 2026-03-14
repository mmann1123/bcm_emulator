"""Teacher forcing ratio scheduling for PCK(t-1) and AET(t-1)."""


def get_tf_ratio(epoch: int, total_epochs: int, warmup_fraction: float = 0.5) -> float:
    """Compute teacher forcing ratio for a given epoch.

    Schedule:
    - Epochs 0 to warmup_end: tf_ratio = 1.0 (100% ground truth)
    - Epochs warmup_end to total_epochs: linearly ramp from 1.0 to 0.0

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of training epochs.
    warmup_fraction : float
        Fraction of epochs with 100% GT (default 0.5).

    Returns
    -------
    float
        Teacher forcing ratio in [0.0, 1.0].
    """
    warmup_end = int(total_epochs * warmup_fraction)

    if epoch < warmup_end:
        return 1.0

    # Linear ramp from 1.0 to 0.0
    ramp_epochs = total_epochs - warmup_end
    if ramp_epochs <= 0:
        return 0.0

    progress = (epoch - warmup_end) / ramp_epochs
    return max(0.0, 1.0 - progress)
