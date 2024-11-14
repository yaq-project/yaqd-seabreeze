"""
seabreeze-script modules are similar to yaqd-ni-daqmx-tmux scripts,
but one difference is that this script explicitly declares the channels
and their properties
"""

import numpy as np


channel_names = ["mean", "a", "b", "d_ba"]
channel_units = {k: "counts" for k in channel_names}
channel_mappings = {k: "wavelength" for k in channel_names}


# --- configure chopping --------------------------------------------------------------------------

chop_index: int = 393  # index used to extract phase
# TODO: index used to reference fluctuations (optional)
reference_index = None
# cutoff (raw counts that distinguish between on and off)
# use "mean" to dynamically extract phase with a mean cutoff
# use "extrema" to dynamically extract phase halfway between min and max values
# TODO: an extraction that works well with bimodal distribution
chop_threshold = "mean"

# -------------------------------------------------------------------------------------------------


sel = (slice(None), chop_index)
ref = (slice(None), reference_index)
if chop_threshold == "mean":
    thresholder = lambda x: x[sel] < x[sel].mean()
elif chop_threshold == "extrema":
    thresholder = lambda x: x[sel] < 0.5 * (x[sel].min() + x[sel].max())
elif chop_threshold in [int, float]:
    thresholder = lambda x: x[sel] < chop_threshold


def process(raw: np.array) -> dict:
    """
    take spectra and extract a chop phase by inspecting at a certain color.
    relies on measuring at some color some scatter that indicates chopper phase
    for proper extaction, we assume chopping phase is pure (i.e. one shot per spectrum)
    e.g. use fastest acquisition time (3 ms), and 1/3 kHz rep rate on fs table
    """
    out = {}
    out["mean"] = raw.mean(axis=0)

    # case chop_threshold
    chop = thresholder(raw)[:, None]

    out["a"] = (raw * chop).sum(axis=0) / chop.sum()
    out["b"] = (raw * ~chop).sum(axis=0) / (~chop).sum()
    out["d_ba"] = out["b"] - out["a"]

    return out
