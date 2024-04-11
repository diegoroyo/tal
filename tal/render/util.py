from dataclasses import dataclass

import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def to_numpy(self):
        return np.array([self.x, self.y])

    @classmethod
    def from_numpy(cls, arr):
        return cls(x=arr[0], y=arr[1])

    # Needed to support unpacking from a `Point2D` instance
    def __iter__(self):
        return iter((self.x, self.y))


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, arr):
        return cls(x=arr[0], y=arr[1], z=arr[2])

    def __iter__(self):
        return iter((self.x, self.y, self.z))


def expand(vec, x, y):
    assert len(vec) == 3
    return vec.reshape(1, 1, 3).repeat(x, axis=0).repeat(y, axis=1)


def import_mitsuba_backend():
    from tal.config import Config, ask_for_config

    mitsuba_version = ask_for_config(Config.MITSUBA_VERSION, force_ask=False)
    if mitsuba_version == "2":
        import tal.render.mitsuba2_transient_nlos as mitsuba_backend
    elif mitsuba_version == "3":
        import tal.render.mitsuba3_transient_nlos as mitsuba_backend
    else:
        raise AssertionError(
            f"Invalid MITSUBA_VERSION={mitsuba_version}, must be one of (2, 3)"
        )
    return mitsuba_backend
