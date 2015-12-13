import pyfftw


class FFTW:
    def __init__(self, shape, dtype, threads=1, inverse=False):
        self.threads = threads

        if inverse:
            self.arr_in = pyfftw.n_byte_align_empty(shape[0], pyfftw.simd_alignment, dtype=dtype)
            self.fftw = pyfftw.builders.irfft2(self.arr_in, shape[1],
                                               threads=threads, avoid_copy=True,
                                               planner_effort='FFTW_EXHAUSTIVE')
        else:
            self.arr_in = pyfftw.n_byte_align_empty(shape, pyfftw.simd_alignment, dtype=dtype)
            self.fftw = pyfftw.builders.rfft2(self.arr_in, overwrite_input=True,
                                              threads=threads, avoid_copy=True,
                                              planner_effort='FFTW_EXHAUSTIVE')

    def get_inverse(self):
        arr_out = self.fftw.get_output_array()
        return FFTW(shape=(arr_out.shape, self.arr_in.shape), dtype=arr_out.dtype, threads=self.threads, inverse=True)

    def __call__(self, arr, reverse=False):
        step = -1 if reverse else 1
        self.arr_in.fill(0)
        self.arr_in[:arr.shape[0], :arr.shape[1]] = arr[::step, ::step]
        return self.fftw().copy()
