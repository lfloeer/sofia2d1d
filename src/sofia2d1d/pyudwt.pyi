from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BOUNDARY_MODES: dict[str, int]
b3spline: NDArray[np.float32]

def openmp_enabled() -> bool: ...
def set_num_threads(n: int) -> None: ...
def get_num_threads() -> int: ...

class WaveletDecomposition2D1D:
    work: NDArray[np.float32]
    data: NDArray[np.float32]
    xy_mother_function: NDArray[np.float32]
    z_mother_function: NDArray[np.float32]
    xy_scales: int
    z_scales: int
    boundary: Literal["periodic", "reflect", "zero"]
    def __init__(self, data: ArrayLike, xy_scales: int = -1, z_scales: int = -1) -> None: ...
    def decompose(self) -> None: ...
    def handle_coefficients(self, work_array: int, xy_scale: int, z_scale: int) -> None: ...

class Denoise2D1DHard(WaveletDecomposition2D1D):
    reconstruction: NDArray[np.float32]
    thresholds: NDArray[np.float32]
    total_power: bool
    xy_approx: bool
    z_approx: bool
    positivity: bool
    def __init__(
        self,
        data: ArrayLike,
        total_power: bool = False,
        xy_approx: bool = False,
        z_approx: bool = False,
        xy_scales: int = -1,
        z_scales: int = -1,
    ) -> None: ...

class Denoise2D1DSoft(Denoise2D1DHard): ...

class Denoise2D1DHardMRS(Denoise2D1DHard):
    mrs: NDArray[np.short]
    @property
    def packed_mrs(self) -> NDArray[np.uint8]: ...
    fix_mrs: bool

class Denoise2D1DAdaptiveMRS(Denoise2D1DHardMRS):
    sigma: float
    def __init__(
        self,
        sigma: float,
        data: ArrayLike,
        total_power: bool = False,
        xy_approx: bool = False,
        z_approx: bool = False,
        xy_scales: int = -1,
        z_scales: int = -1,
    ) -> None: ...

class WaveletDecomposition1D:
    @property
    def work(self) -> NDArray[np.float32]: ...
    @property
    def scales(self) -> int: ...
    @property
    def data(self) -> NDArray[np.float32]: ...
    mother_function: NDArray[np.float32]
    def __init__(self, data: ArrayLike, scales: int = -1) -> None: ...
    def decompose(self) -> None: ...
    def handle_coefficients(self, work_array: int, scale: int) -> None: ...

class Denoise1DHardMRS(WaveletDecomposition1D):
    reconstruction: NDArray[np.float32]
    thresholds: NDArray[np.float32]
    total_power: bool
    fix_mrs: bool
    @property
    def mrs(self) -> NDArray[np.short]: ...
    def __init__(
        self, data: ArrayLike, thresholds: ArrayLike, scales: int = -1, total_power: bool = False
    ) -> None: ...

class WaveletDecomposition3D:
    @property
    def work(self) -> NDArray[np.float32]: ...
    @property
    def scales(self) -> int: ...
    @property
    def data(self) -> NDArray[np.float32]: ...
    mother_function: NDArray[np.float32]
    def __init__(self, data: ArrayLike, scales: int = -1) -> None: ...
    def decompose(self) -> None: ...
    def handle_coefficients(self, work_array: int, scale: int) -> None: ...
