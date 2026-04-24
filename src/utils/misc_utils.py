from typing import Any, List, Tuple, Union
import numpy as np

np.set_printoptions(precision=5, suppress=True)


class DummyClass:
    def __getattr__(self, name):
        # Return another DummyClass for chained attribute access
        return (
            DummyClass()
        )  # lambda *args, **kwargs: None  # no-op callable placeholder

    def __call__(self, *args, **kwargs):
        # Callable no-op (e.g. canvas.ax.scatter())
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def color_string(text, font_color, background_color=None):
    """Construct a string with colored output, by Xuechao
    font color: 'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
    background color: (optional) same options as font color
    """
    font_color_dict = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
        "white": 37,
    }
    background_color_dict = {
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "purple": 45,
        "cyan": 46,
        "white": 47,
    }

    if font_color not in font_color_dict:
        return text  # fallback if invalid color

    font_code = font_color_dict[font_color]
    if background_color is None:
        return f"\033[1;{font_code}m{text}\033[0m"
    else:
        bg_code = background_color_dict.get(background_color, 40)
        return f"\033[1;{font_code};{bg_code}m{text}\033[0m"
