import numpy as np
from planfftw import correlate


def _init_mask(template, mask):
    if mask is None:
        mask = np.ones_like(template, dtype=bool)
    mask = mask.astype(np.float32)
    return mask


def wncc(image, template, mask=None):
    """
    Computes the Weighted Normalized Cross Correlation.

    :param image: First array - image.
    :param template: Second array - template.
    :param mask: Mask containing weights for template pixels. Same size as template.
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
    $$= \sum f(x, y)^2 m(x-u, y-v) + \bar f(u, v)^2 \sum m(x-u, y-v) - 2 \bar f(u, v) \sum f(x, y) m(x-u, y-v)$$
    $$denom_2(u, v) = \sum [t(x, y) - \bar t]^2 m(x, y).$$
    """
    mask = _init_mask(template, mask)

    image_corr = correlate(image, template.shape, constant_x=True)
    image2_corr = correlate(image ** 2, template.shape, constant_x=True)

    image_corr_mask = image_corr(mask)

    bar_t = (template * mask).sum() / mask.sum()
    bar_f = image_corr_mask / mask.sum()

    nom = image_corr((template - bar_t) * mask)

    denom1 = image2_corr(mask) + bar_f ** 2 * mask.sum() - 2 * bar_f * image_corr_mask

    denom2 = ((template - bar_t) ** 2 * mask).sum()

    result = nom / np.sqrt(denom1 * denom2)

    result[np.abs(denom1) < 1e-15] = float('nan')
    if np.abs(denom2) < 1e-15:
        result[...] = float('nan')

    return result


def _wncc_fix_image(image, template_shape):
    image_corr = correlate(image, template_shape, constant_x=True)
    image2_corr = correlate(image ** 2, template_shape, constant_x=True)

    def ncc_fixed_image(template, mask=None):
        mask = _init_mask(template, mask)

        image_corr_mask = image_corr(mask)

        bar_t = (template * mask).sum() / mask.sum()
        bar_f = image_corr_mask / mask.sum()

        nom = image_corr((template - bar_t) * mask)
        denom1 = image2_corr(mask) + bar_f ** 2 * mask.sum() - 2 * bar_f * image_corr_mask
        denom2 = ((template - bar_t) ** 2 * mask).sum()

        result = nom / np.sqrt(denom1 * denom2)

        result[np.abs(denom1) < 1e-15] = float('nan')
        if np.abs(denom2) < 1e-15:
            result[...] = float('nan')

        return result

    return ncc_fixed_image


def _wncc_fix_template(image_shape, template, mask=None):
    mask = _init_mask(template, mask)

    bar_t = (template * mask).sum() / mask.sum()

    denom2 = ((template - bar_t) ** 2 * mask).sum()

    corr_mask = correlate(image_shape, mask, constant_y=True)
    corr_tmplmask = correlate(image_shape, (template - bar_t) * mask, constant_y=True)

    def ncc_fixed_template(image):
        image_corr_mask = corr_mask(image)
        bar_f = image_corr_mask / mask.sum()

        nom = corr_tmplmask(image)
        denom1 = corr_mask(image ** 2) + bar_f ** 2 * mask.sum() - 2 * bar_f * image_corr_mask

        result = nom / np.sqrt(denom1 * denom2)

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
