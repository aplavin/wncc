import numpy as np
from .fft import FFTW


def _next_regular(target):
    """
    Copied from scipy.

    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & ( target -1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


class Convolver:
    def __init__(self, shape_first, shape_second, dtype, threads=1):
        self.shape_first = shape_first
        self.shape_second = shape_second

        self.dtype = np.dtype(dtype)

        self.shape_result = np.array(self.shape_first) + self.shape_second - 1
        self.slice_result = tuple([slice(0, int(d)) for d in self.shape_result])
        self.shape_fft = [_next_regular(int(d)) for d in self.shape_result]

        self.fftw = FFTW(shape=self.shape_fft, dtype=self.dtype, threads=threads)
        self.ifftw = self.fftw.get_inverse()

        self.arrays = {}
        self.ffts = {}
        self.convolutions = {}

    def add_array(self, name, arr):
        if name in self.arrays:
            for k in self.arrays[name][1][0]:
                del self.ffts[k]
            for k in self.arrays[name][1][1]:
                del self.convolutions[k]
        assert arr.dtype == self.dtype
        assert arr.shape in (self.shape_first, self.shape_second)
        self.arrays[name] = (arr, ([], []))

    def _fft(self, name, reverse):
        key = (name, reverse)
        if key not in self.ffts:
            self.ffts[key] = self.fftw(self.arrays[name][0], reverse)
            self.arrays[name][1][0].append(key)
        return self.ffts[key]

    def _convolve(self, name_first, name_second, correlate):
        key = (name_first, name_second, correlate)
        if key not in self.convolutions:
            fft = self._fft(name_first, reverse=False) * self._fft(name_second, reverse=correlate)
            ret = self.ifftw(fft)
            self.convolutions[key] = ret[self.slice_result].copy()

            self.arrays[name_first][1][1].append(key)
            self.arrays[name_second][1][1].append(key)
        return self.convolutions[key]

    def convolve(self, name_first, name_second):
        return self._convolve(name_first, name_second, False)

    def correlate(self, name_first, name_second):
        return self._convolve(name_first, name_second, True)
