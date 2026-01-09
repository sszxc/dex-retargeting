import time
import numpy as np
from rich import print


class Timer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.times = {}
        self.start()

    def start(self):
        self.last_time = time.perf_counter()
        self.module_start_time = self.last_time

    def reset(self):
        self.times = {}
        self.start()

    def check(self, name: str):
        if not self.enabled:
            return
        _time = time.perf_counter()
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(_time - self.last_time)
        self.last_time = _time

    def get_time(self, name: str) -> dict:
        """return a dict of time statistics"""
        if name not in self.times:
            return {}
        return {
            "count": len(self.times[name]),
            "mean": sum(self.times[name]) / len(self.times[name]),
            "var": float(np.var(self.times[name])),
            "max": max(self.times[name]),
            "min": min(self.times[name]),
        }

    def print_times_statistics(self):
        if not self.enabled:
            return
        for name in self.times:
            print(name, self.get_time(name))
        print(f"Total time: {time.perf_counter() - self.module_start_time:.6f}s")

    def print_times(self):
        if not self.enabled:
            return
        _max_name_len = max(len(name) for name in self.times)
        for name in self.times:
            print(
                f"{name:<{_max_name_len}}: {self.get_time(name)['mean']:.6f}s * {self.get_time(name)['count']}"
            )
        print(
            f"{'Total time':<{_max_name_len}}: {time.perf_counter() - self.module_start_time:.6f}s"
        )
