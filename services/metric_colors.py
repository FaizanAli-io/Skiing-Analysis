"""Shared BlueIQ metric colors for video vectors and score-panel bars."""

METRIC_COLORS_RGB = {
    "edging": (106, 175, 255),
    "pressure": (34, 197, 94),
    "rotation": (249, 115, 22),
    "balance": (245, 158, 11),
}

METRIC_COLORS_BGR = {
    metric: (color[2], color[1], color[0])
    for metric, color in METRIC_COLORS_RGB.items()
}
