import hypothesis as h
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from .fft import FFTW

st_myfloats = st.floats(1e-5, 1e5) | st.just(0) | st.floats(-1e5, -1e-5)
st_arr_shape = st.tuples(st.integers(2, 5), st.integers(2, 5)) \
    .flatmap(lambda shape: st.tuples(npst.arrays(float, shape, st_myfloats),
                                     st.tuples(st.integers(shape[0], 6), st.integers(shape[1], 6))))


@h.given(st_arr_shape)
def test_fft(arr_shape):
    arr, shape = arr_shape

    fftw = FFTW(shape, arr.dtype)
    expected = np.fft.rfft2(arr, shape)
    result = fftw(arr)

    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, equal_nan=True)


st_arr_arr_shape = st.tuples(st.integers(2, 5), st.integers(2, 5)) \
    .flatmap(lambda shape: st.tuples(npst.arrays(float, shape, st_myfloats),
                                     npst.arrays(float, [s - 1 for s in shape], st_myfloats),
                                     st.tuples(st.integers(shape[0], 6), st.integers(shape[1], 6))))


@h.given(st_arr_arr_shape)
def test_fft_multiple(arr_arr_shape):
    arr1, arr2, shape = arr_arr_shape

    fftw = FFTW(shape, arr1.dtype)
    expected = np.fft.rfft2(arr1, shape)
    result = fftw(arr1)

    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, equal_nan=True)

    expected = np.fft.rfft2(arr2, shape)
    result = fftw(arr2)

    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, equal_nan=True)


@h.given(st_arr_shape)
def test_inverse_fft(arr_shape):
    arr, shape = arr_shape

    fft = np.fft.rfft2(arr, shape)

    fftw = FFTW(shape, arr.dtype)
    ifftw = fftw.get_inverse()
    expected = np.fft.irfft2(fft, shape)
    result = ifftw(fft)

    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, equal_nan=True)
