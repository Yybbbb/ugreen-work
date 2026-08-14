import math


def rgb_to_lab(rgb):
    if _is_pixel(rgb):
        return _rgb_pixel_to_lab(rgb)
    return [rgb_to_lab(item) for item in rgb]


def chroma_ab(lab):
    if _is_lab_pixel(lab):
        return math.sqrt(lab[1] * lab[1] + lab[2] * lab[2])
    return [chroma_ab(item) for item in lab]


def _is_pixel(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, int) for v in value)
    )


def _is_lab_pixel(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _rgb_pixel_to_lab(pixel):
    r, g, b = [channel / 255.0 for channel in pixel]
    r, g, b = [_srgb_to_linear(v) for v in (r, g, b)]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    z /= 1.08883

    fx = _xyz_f(x)
    fy = _xyz_f(y)
    fz = _xyz_f(z)

    return [
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    ]


def _srgb_to_linear(v):
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def _xyz_f(t):
    if t > 0.008856:
        return t ** (1.0 / 3.0)
    return 7.787 * t + 16.0 / 116.0
