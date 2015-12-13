import numpy as np
import numexpr as ne
from .convolve import Convolver


def _init_mask(template, mask):
    if mask is None:
        mask = np.ones_like(template)
    return mask.astype(template.dtype)


def wncc(image, template, mask=None, threads=1):
    """
    Computes the Weighted Normalized Cross Correlation.

    :param image: First array - image.
    :param template: Second array - template.
    :param mask: Mask containing weights for template pixels. Same size as template.
    :param threads: Number of threads (passed to FFTW).
    :return: The resulting cross correlation.

    This function returns the result of computing the formulae
    (here f, t and m denote image, template and mask correspondingly):
    $$wncc(u, v) = \frac{nom(u, v)}{\sqrt{denom_1(u, v) denom_2(u, v)}}$$
    $$nom(u, v) = \sum [f(x,y) - \bar f(u, v)]  [t(x-u, y-v) - \bar t] m(x-u, y-v)$$
    $$denom_1(u, v) = \sum [f(x, y) - \bar f(u, v)]^2 m(x-u, y-v)$$
    $$denom_2(u, v) = \sum [t(x-u, y-v) - \bar t]^2 m(x-u, y-v)$$
    $$\bar f(u, v) = \frac{\sum f(x, y) m(x-u, y-v)}{\sum m(x, y)}$$
    $$\bar t = \frac{\sum t(x, y)  m(x, y)}{\sum m(x, y)}.$$

    The computations are done using FFT convolution instead of naive summation,
    which is possible due to such transformations:
    $$nom(u, v) = \sum f(x,y) [t(x-u, y-v) - \bar t] m(x-u, y-v)$$
    $$denom_1(u, v) = \sum [f(x, y)^2 + \bar f(u, v)^2 - 2 f(x, y) \bar f(u, v)] m(x-u, y-v) =$$
    $$= \sum f(x, y)^2 m(x-u, y-v) + \bar f(u, v)^2 \sum m(x-u, y-v) - 2 \bar f(u, v) \sum f(x, y) m(x-u, y-v) =$$
    $$= \sum f(x, y)^2 m(x-u, y-v) - \bar f(u, v)^2 \sum m(x, y) =$$
    $$= \sum f(x, y)^2 m(x-u, y-v) - \frac{\left(\sum f(x, y) m(x-u, y-v)\right)^2}{\sum m(x, y)}$$
    $$denom_2(u, v) = \sum [t(x, y) - \bar t]^2 m(x, y).$$
    """
    mask = _init_mask(template, mask)

    mask_sum = ne.evaluate('sum(mask)')
    template_mask_sum = ne.evaluate('sum(template * mask)')
    bar_t = template_mask_sum / mask_sum

    convolver = Convolver(image.shape, template.shape, threads=threads, dtype=image.dtype)
    convolver.add_array('image', image)
    convolver.add_array('image_sq', ne.evaluate('image ** 2'))
    convolver.add_array('mask', mask)
    convolver.add_array('tmpl_mask', ne.evaluate('(template - bar_t) * mask'))

    nom = convolver.correlate('image', 'tmpl_mask')
    denom1 = ne.evaluate('a - b**2 / mask_sum', global_dict={'a': convolver.correlate('image_sq', 'mask'),
                                                             'b': convolver.correlate('image', 'mask')})
    denom2 = ne.evaluate('sum((template - bar_t) ** 2 * mask)')

    result = ne.evaluate('nom / sqrt(denom1 * denom2)')

    result[np.abs(denom1) < 1e-15] = float('nan')
    if np.abs(denom2) < 1e-15:
        result[...] = float('nan')

    return result


def _wncc_fix_image(image, template_shape, threads=1):
    convolver = Convolver(image.shape, template_shape, threads=threads, dtype=image.dtype)
    convolver.add_array('image', image)
    convolver.add_array('image_sq', ne.evaluate('image ** 2'))

    def ncc_fixed_image(template, mask=None):
        mask = _init_mask(template, mask)

        mask_sum = ne.evaluate('sum(mask)')
        template_mask_sum = ne.evaluate('sum(template * mask)')
        bar_t = template_mask_sum / mask_sum

        convolver.add_array('mask', mask)
        convolver.add_array('tmpl_mask', ne.evaluate('(template - bar_t) * mask'))

        nom = convolver.correlate('image', 'tmpl_mask')
        denom1 = ne.evaluate('a - b**2 / mask_sum', global_dict={'a': convolver.correlate('image_sq', 'mask'),
                                                                 'b': convolver.correlate('image', 'mask')})
        denom2 = ne.evaluate('sum((template - bar_t) ** 2 * mask)')

        result = ne.evaluate('nom / sqrt(denom1 * denom2)')

        result[np.abs(denom1) < 1e-15] = float('nan')
        if np.abs(denom2) < 1e-15:
            result[...] = float('nan')

        return result

    return ncc_fixed_image


def _wncc_fix_template(image_shape, template, mask=None, threads=1):
    mask = _init_mask(template, mask)

    mask_sum = ne.evaluate('sum(mask)')
    template_mask_sum = ne.evaluate('sum(template * mask)')
    bar_t = template_mask_sum / mask_sum

    denom2 = ne.evaluate('sum((template - bar_t) ** 2 * mask)')

    convolver = Convolver(image_shape, template.shape, threads=threads, dtype=template.dtype)
    convolver.add_array('mask', mask)
    convolver.add_array('tmpl_mask', ne.evaluate('(template - bar_t) * mask'))

    def ncc_fixed_template(image):
        convolver.add_array('image', image)
        convolver.add_array('image_sq', ne.evaluate('image ** 2'))

        nom = convolver.correlate('image', 'tmpl_mask')
        denom1 = ne.evaluate('a - b**2 / mask_sum', global_dict={'a': convolver.correlate('image_sq', 'mask'),
                                                                 'b': convolver.correlate('image', 'mask'),
                                                                 'mask_sum': mask_sum})

        result = ne.evaluate('nom / sqrt(denom1 * denom2)')

        result[np.abs(denom1) < 1e-15] = float('nan')
        if np.abs(denom2) < 1e-15:
            result[...] = float('nan')

        return result

    return ncc_fixed_template


def wncc_prepare(image=None, template=None, mask=None):
    if isinstance(image, np.ndarray):
        assert mask is None
        assert isinstance(template, tuple)
        return _wncc_fix_image(image, template)
    elif isinstance(template, np.ndarray):
        assert isinstance(image, tuple)
        return _wncc_fix_template(image, template, mask)
    else:
        raise ValueError('Neither image nor template are numpy arrays.')


def _wncc_naive(f, t, m, return_func=False):
    """
    Naive implementation of the following formulae:
    $$wncc(u, v) = \frac{nom(u, v)}{\sqrt{denom_1(u, v) denom_2(u, v)}}$$
    $$nom(u, v) = \sum [f(x,y) - \bar f(u, v)]  [t(x-u, y-v) - \bar t] m(x-u, y-v)$$
    $$denom_1(u, v) = \sum [f(x, y) - \bar f(u, v)]^2 m(x-u, y-v)$$
    $$denom_2(u, v) = \sum [t(x-u, y-v) - \bar t]^2 m(x-u, y-v)$$
    $$\bar f(u, v) = \frac{\sum f(x, y) m(x-u, y-v)}{\sum m(x, y)}$$
    $$\bar t = \frac{\sum t(x, y)  m(x, y)}{\sum m(x, y)}$$
    """
    assert t.shape == m.shape

    def f_(x, y):
        if 0 <= x < f.shape[0] and 0 <= y < f.shape[1]:
            return f[x, y]
        return 0

    def at(u, v):
        f_bar = sum(f_(x, y) * m[x - u, y - v]
                    for x in range(u, u + m.shape[0])
                    for y in range(v, v + m.shape[1])) \
                / m.sum()
        t_bar = sum(t[x, y] * m[x, y]
                    for x in range(t.shape[0])
                    for y in range(t.shape[1])) \
                / m.sum()
        nom = sum((f_(x, y) - f_bar) * (t[x - u, y - v] - t_bar) * m[x - u, y - v]
                  for x in range(u, u + m.shape[0])
                  for y in range(v, v + m.shape[1]))
        denom_1 = sum((f_(x, y) - f_bar) ** 2 * m[x - u, y - v]
                      for x in range(u, u + m.shape[0])
                      for y in range(v, v + m.shape[1]))
        denom_2 = sum((t[x, y] - t_bar) ** 2 * m[x, y]
                      for x in range(0, t.shape[0])
                      for y in range(0, t.shape[1]))
        if abs(denom_1) < 1e-15 or abs(denom_2) < 1e-15:
            return float('nan')
        return nom / np.sqrt(denom_1 * denom_2)

    if return_func:
        return at
    else:
        result = [[at(u, v)
                   for v in range(-t.shape[1] + 1, f.shape[1])]
                  for u in range(-t.shape[0] + 1, f.shape[0])]
        return np.array(result)