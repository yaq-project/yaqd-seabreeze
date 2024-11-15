"""
take spectra and extract a chop phase by inspecting at a certain color.
relies on measuring, at some color, light that switches with chopper phase
for proper extaction, we assume chopping phase is pure (i.e. one shot per spectrum).
e.g. use fastest acquisition time (3 ms), and 1/3 kHz rep rate on fs table

seabreeze-script modules are similar to yaqd-ni-daqmx-tmux scripts.
one difference: these scripts explicitly declares the channels and some properties
"""

import numpy as np


# --- configure -----------------------------------------------------------------------------
# --- --- channels --------------------------------------------------------------------------

channel_names = ["mean", "a", "b", "d_ba"]
channel_units = {k: "counts" for k in channel_names}
channel_mappings = {k: "wavelength" for k in channel_names}

# --- --- discriminator ---------------------------------------------------------------------

discriminator_index = 631  # discriminate spurious 2-shot acquisitions; use None to turn off
discriminator_limits = [0.3, 1.7]  # signals larger than 1.7x median value are removed

# --- --- chopping --------------------------------------------------------------------------

chop_index: int = 360  # index used to extract phase

# cutoff (raw counts that distinguish between on and off)
# use "mean" to dynamically extract phase with a mean cutoff
# use "extrema" to dynamically extract phase halfway between min and max values
chop_threshold = "extrema"

# -------------------------------------------------------------------------------------------------


sel = (slice(None), chop_index)
des = (slice(None), discriminator_index)


def discriminator(x):
    xdes = x[des] / np.median(x[des])
    return (xdes > discriminator_limits[0]) & (xdes < discriminator_limits[1])


if chop_threshold == "mean":
    thresholder = lambda x: x[sel] < x[sel].mean()
elif chop_threshold == "extrema":
    thresholder = lambda x: x[sel] < 0.5 * (x[sel].min() + x[sel].max())
elif chop_threshold in [int, float]:
    thresholder = lambda x: x[sel] < chop_threshold


def process(raw: np.array) -> dict:
    out = {}

    if discriminator_index is not None:
        valid = discriminator(raw)
        # TODO: make a channel to alert when discriminator fails
        if valid.any():
            raw = raw[valid]

    out["mean"] = (raw).mean(axis=0)

    chop = thresholder(raw)

    out["a"] = raw[chop].mean(axis=0)
    out["b"] = raw[~chop].mean(axis=0)
    out["d_ba"] = out["b"] - out["a"]

    return out
