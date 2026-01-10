from typing import Any, List, Tuple, Union
import numpy as np

np.set_printoptions(precision=5, suppress=True)


class DummyClass:
    def __getattr__(self, name):
        # 返回另一个 DummyClass 实例，从而支持链式调用
        return (
            DummyClass()
        )  # lambda *args, **kwargs: None  # 返回一个直接 pass 的空函数

    def __call__(self, *args, **kwargs):
        # 允许 DummyClass 实例被调用（如 canvas.ax.scatter()），直接 pass
        return None

    def __enter__(self):
        # 用于进入 with 语句
        return self

    def __exit__(self, *args):
        # 用于退出 with 语句
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
