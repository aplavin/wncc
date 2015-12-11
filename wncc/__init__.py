import numpy as np
from planfftw import correlate


def _init_mask(template, mask):
    if mask is None:
        mask = np.ones_like(template, dtype=bool)
    mask = mask.astype(np.float32)
    return mask


def wncc(image, template, mask=None):
    mask = _init_mask(template, mask)

    image_corr = correlate(image, template.shape, constant_x=True)
    image2_corr = correlate(image ** 2, template.shape, constant_x=True)

    image_corr_mask = image_corr(mask)

    bar_t = (template * mask).sum() / mask.sum()
    bar_f = image_corr_mask / mask.sum()

    nom = image_corr((template - bar_t) * mask)

    denom1 = image2_corr(mask) + bar_f ** 2 * mask.sum() - 2 * bar_f * image_corr_mask

    denom2 = ((template - bar_t) ** 2 * mask).sum()
    assert denom2 != 0

    result = nom / np.sqrt(denom1 * denom2)
    result[denom1 == 0] = float('nan')

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
        assert denom2 != 0

        result = nom / np.sqrt(denom1 * denom2)
        result[denom1 == 0] = float('nan')

        return result

    return ncc_fixed_image


def _wncc_fix_template(image_shape, template, mask=None):
    mask = _init_mask(template, mask)

    bar_t = (template * mask).sum() / mask.sum()

    denom2 = ((template - bar_t) ** 2 * mask).sum()
    assert denom2 != 0

    corr_mask = correlate(image_shape, mask, constant_y=True)
    corr_tmplmask = correlate(image_shape, (template - bar_t) * mask, constant_y=True)

    def ncc_fixed_template(image):
        image_corr_mask = corr_mask(image)
        bar_f = image_corr_mask / mask.sum()

        nom = corr_tmplmask(image)
        denom1 = corr_mask(image ** 2) + bar_f ** 2 * mask.sum() - 2 * bar_f * image_corr_mask

        result = nom / np.sqrt(denom1 * denom2)
        result[denom1 == 0] = float('nan')

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
