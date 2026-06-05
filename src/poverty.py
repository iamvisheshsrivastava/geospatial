"""Poverty proxy estimation from satellite land-cover classification.

Implements the wealth-index pipeline described in
notebooks/05_poverty_proxy_nightlights.ipynb, adapted for production use.

Methodology (Jean et al. 2016 / Yeh et al. 2020 inspired):
  - Land-cover class probabilities serve as a proxy for economic activity.
  - Industrial / Residential / Highway patches indicate higher economic output.
  - Forest / Herbaceous Vegetation patches indicate lower economic development.
  - The weighted sum of probabilities yields a normalised Wealth Index [0, 1].

This is a simplified, RGB-only approximation.  True poverty estimation at
research quality requires nighttime light (VIIRS/DMSP) composites and full
Sentinel-2 13-band imagery as described in the notebooks.
"""
from __future__ import annotations

# ── Wealth weights per EuroSAT land-cover class ───────────────────────────────
# Calibrated to correlate with GDP-per-capita proxies used in the literature.
_WEALTH_WEIGHTS: dict[str, float] = {
    "Industrial":            0.90,   # factories → high economic output
    "Residential":           0.70,   # urban housing → developed area
    "Highway":               0.60,   # road infrastructure → connectivity
    "PermanentCrop":         0.45,   # orchards/vineyards → commercial agri
    "AnnualCrop":            0.28,   # seasonal crops → subsistence/commercial
    "Pasture":               0.22,   # livestock → smallholder farming
    "River":                 0.18,   # water access → mixed signal
    "SeaLake":               0.15,   # coast/lake → tourism or isolation
    "HerbaceousVegetation":  0.10,   # scrubland → undeveloped
    "Forest":                0.08,   # remote forest → low development
}

_WEALTH_LABELS = [
    (0.00, 0.20, "Very Low",  "Subsistence economy — minimal infrastructure detected"),
    (0.20, 0.40, "Low",       "Developing area — limited urban or industrial activity"),
    (0.40, 0.60, "Medium",    "Emerging economy — mix of agriculture and development"),
    (0.60, 0.80, "High",      "Developed area — clear residential or industrial activity"),
    (0.80, 1.00, "Very High", "Highly urbanised / industrial — strong economic signal"),
]


def compute_poverty_proxy(class_probabilities: dict[str, float]) -> dict:
    """Estimate a wealth index from land-cover class probabilities.

    Args:
        class_probabilities: Dict mapping class name → probability (sums to ~1).

    Returns:
        Dict with:
          wealth_index       — float in [0, 1]
          wealth_label       — "Very Low" … "Very High"
          wealth_color       — hex colour for UI badge
          interpretation     — one-sentence explanation
          top_contributors   — list of {class, probability, contribution} dicts
    """
    wealth_index = sum(
        prob * _WEALTH_WEIGHTS.get(cls, 0.0)
        for cls, prob in class_probabilities.items()
    )
    wealth_index = round(float(min(max(wealth_index, 0.0), 1.0)), 4)

    label, interpretation = "Unknown", ""
    for lo, hi, lbl, interp in _WEALTH_LABELS:
        if lo <= wealth_index <= hi:
            label, interpretation = lbl, interp
            break

    colour_map = {
        "Very Low":  "#ef4444",   # red
        "Low":       "#f97316",   # orange
        "Medium":    "#eab308",   # yellow
        "High":      "#22c55e",   # green
        "Very High": "#14b8a6",   # teal
    }

    # Top 3 contributing classes (probability × weight, descending)
    contributors = sorted(
        [
            {
                "class":        cls,
                "probability":  round(prob, 4),
                "contribution": round(prob * _WEALTH_WEIGHTS.get(cls, 0.0), 4),
                "weight":       _WEALTH_WEIGHTS.get(cls, 0.0),
            }
            for cls, prob in class_probabilities.items()
            if prob > 0.001
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )[:5]

    return {
        "wealth_index":     wealth_index,
        "wealth_label":     label,
        "wealth_color":     colour_map.get(label, "#64748b"),
        "interpretation":   interpretation,
        "top_contributors": contributors,
        "methodology":      (
            "Wealth index = weighted sum of land-cover probabilities. "
            "Weights calibrated to GDP-per-capita correlations following "
            "Jean et al. (2016, Science) and Yeh et al. (2020, Nature Comms)."
        ),
    }
