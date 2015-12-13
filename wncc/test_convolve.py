import hypothesis as h
import numpy as np
import scipy as sp
import scipy.signal
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from .convolve import Convolver

st_myfloats = st.floats(1e-5, 1e5) | st.just(0) | st.floats(-1e5, -1e-5)
st_arr_arr = st.tuples(st.integers(2, 5), st.integers(2, 5)) \
    .flatmap(lambda shape: st.tuples(npst.arrays(float, shape, st_myfloats),
                                     npst.arrays(float, shape, st_myfloats)))


@h.given(st_arr_arr)
def test_same_shape(arr_arr):
    arr1, arr2 = arr_arr

    c = Convolver(arr1.shape, arr2.shape, arr1.dtype)
    c.add_array('A', arr1)
    c.add_array('B', arr2)

    expected = sp.signal.fftconvolve(arr1, arr2)
    result = c.convolve('A', 'B')
    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, atol=1e-3, equal_nan=True)


@h.given(st_arr_arr)
def test_correlate(arr_arr):
    arr1, arr2 = arr_arr

    c = Convolver(arr1.shape, arr2.shape, arr1.dtype)
    c.add_array('A', arr1)
    c.add_array('B', arr2)

    expected = sp.signal.fftconvolve(arr1, arr2[::-1, ::-1])
    result = c.correlate('A', 'B')
    h.note(expected)
    h.note(result)
    assert np.allclose(expected, result, atol=1e-3, equal_nan=True)
