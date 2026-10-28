"""Band tables for the 32-band OHS imagery of HRSSD.

ENTRY_WAVELENGTHS_NM are the band centres supplied to the wavelength-conditioned DOFA entry in every
experiment of the paper (default band set of the OHS-3 series). The HRSSD / WHU-OHS imagery follows the
nominal OHS-2 band set (IMAGE_BAND_CENTRES_NM); the two tables agree for 16 of the 32 bands
(Appendix A of the paper). Keep ENTRY_WAVELENGTHS_NM to reproduce the reported results.
"""

ENTRY_WAVELENGTHS_NM = [
    443, 466, 490, 500, 510, 531, 550, 560, 580, 596, 620, 640, 665, 670, 686, 700,
    709, 730, 746, 760, 776, 780, 806, 820, 833, 850, 865, 880, 896, 910, 926, 940,
]

IMAGE_BAND_CENTRES_NM = [
    466, 480, 500, 520, 536, 550, 566, 580, 596, 610, 626, 640, 656, 670, 686, 700,
    716, 730, 746, 760, 776, 790, 806, 820, 836, 850, 866, 880, 896, 910, 926, 940,
]

# Three-band reference arm used by the diagnostics: bands 14, 8 and 3 (0-based indices below).
VISIBLE_3B = [13, 7, 2]

# Band configurations of Table A2 (0-based indices), used by KernelScope.
BAND_CONFIGS = {
    "Full OHS (32b)": list(range(32)),
    "Visible 3b": VISIBLE_3B,
    "Random 3b": [10, 11, 22],
    "Spread 3b": [0, 16, 31],
    "S2-like 8b": [0, 2, 7, 12, 16, 18, 25, 31],
    "Adjacent 8b (bands 1-8)": list(range(0, 8)),
    "Adjacent 8b (bands 13-20)": list(range(12, 20)),
    "Base 4b": [0, 12, 18, 31],
    "Twin 8b": [0, 1, 12, 13, 18, 19, 30, 31],
}


def wavelengths_um(idx=None, table=ENTRY_WAVELENGTHS_NM):
    """DOFA expects micrometres."""
    idx = range(len(table)) if idx is None else idx
    return [table[i] / 1000.0 for i in idx]
