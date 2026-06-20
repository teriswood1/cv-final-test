from typing import List, Tuple
import colorsys


def get_color_map(num_classes: int) -> List[Tuple[int, int, int]]:
    """Return a list of RGB tuples for `num_classes` distinct colors."""
    colors = []
    for i in range(num_classes):
        hue = i / max(1, num_classes)
        lightness = 0.5
        saturation = 0.9
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors
