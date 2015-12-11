import numpy as np
import hypothesis as h
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from wncc import wncc, _wncc_naive


@h.given(st.tuples(st.integers(1, 10), st.integers(1, 10)),
         st.tuples(st.integers(1, 10), st.integers(1, 10)))
def test_random(shape_image, shape_template):
    image = np.random.rand(*shape_image)
    template = np.random.rand(*shape_template)
    mask = np.random.rand(*shape_template)
    naive_result = _wncc_naive(image, template, mask)
    result = wncc(image, template, mask)
    assert np.allclose(naive_result, result, atol=1e-3, equal_nan=True)


class GenBasic:
    myfloats = st.integers(-20, 20).map(lambda i: i / 2)
    myfloats_pos = st.integers(0, 20).map(lambda i: i / 2)
    nparrs = st.tuples(st.integers(1, 5), st.integers(1, 5)) \
        .flatmap(lambda shape: npst.arrays(float, shape, GenBasic.myfloats))
    nparrs_two = st.tuples(st.integers(1, 5), st.integers(1, 5)) \
        .flatmap(lambda shape: st.tuples(npst.arrays(float, shape, GenBasic.myfloats),
                                         npst.arrays(float, shape, GenBasic.myfloats_pos)))


@h.given(GenBasic.nparrs, GenBasic.nparrs_two)
def test_gen_basic(image, templatemask):
    template, mask = templatemask

    naive_result = _wncc_naive(image, template, mask)
    result = wncc(image, template, mask)

    h.note(naive_result)
    h.note(result)

    naive_finite = naive_result[np.isfinite(naive_result)]
    naive_infinite = naive_result[~np.isfinite(naive_result)]
    res_finite = result[np.isfinite(naive_result)]
    res_infinite = result[~np.isfinite(naive_result)]

    assert np.allclose(naive_finite, res_finite)
    assert np.isnan(naive_infinite).all()
    assert (np.isnan(res_infinite) | np.isclose(res_infinite, 0, atol=1e-5)).all()


class GenRandom:
    myfloats = st.sampled_from((np.random.rand(50) - 0.5) * 20)
    myfloats_pos = st.sampled_from(np.random.rand(50) * 10)
    nparrs = st.tuples(st.integers(1, 5), st.integers(1, 5)) \
        .flatmap(lambda shape: npst.arrays(float, shape, GenRandom.myfloats))
    nparrs_two = st.tuples(st.integers(1, 5), st.integers(1, 5)) \
        .flatmap(lambda shape: st.tuples(npst.arrays(float, shape, GenRandom.myfloats),
                                         npst.arrays(float, shape, GenRandom.myfloats_pos)))


@h.given(GenRandom.nparrs, GenRandom.nparrs_two)
def test_gen_random(image, templatemask):
    template, mask = templatemask

    naive_result = _wncc_naive(image, template, mask)
    result = wncc(image, template, mask)

    h.note(naive_result)
    h.note(result)

    naive_finite = naive_result[np.isfinite(naive_result) & np.isfinite(result)]
    res_finite = result[np.isfinite(naive_result) & np.isfinite(result)]

    assert np.allclose(naive_finite, res_finite, atol=1e-3)