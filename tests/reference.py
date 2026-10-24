"""
Straightforward NumPy implementation of the 2D-1D transform and de-noising,
following the original implementation operation by operation (single
precision, same order of summation). It is slow and memory hungry, but easy to
check, and the optimised code must reproduce it exactly.
"""

import numpy as np

B3SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / np.float32(16.0)


def _indices(n, offset, mode):
    idx = np.arange(n) + offset
    if mode == "periodic":
        return idx % n, None
    if mode == "reflect":
        if n == 1:
            return np.zeros(n, dtype=int), None
        period = 2 * (n - 1)
        idx = idx % period
        return np.where(idx >= n, period - idx, idx), None
    valid = (idx >= 0) & (idx < n)
    return np.clip(idx, 0, n - 1), valid


def smooth(a, kernel, step, axis, mode="periodic"):
    """A trous convolution of a along axis."""
    a = np.asarray(a, dtype=np.float32)
    n = a.shape[axis]
    h = len(kernel) // 2
    out = np.zeros_like(a)
    for j, k in enumerate(kernel):
        idx, valid = _indices(n, (j - h) * step, mode)
        term = np.take(a, idx, axis=axis) * np.float32(k)
        if valid is not None:
            shape = [1] * a.ndim
            shape[axis] = n
            term = term * valid.reshape(shape)
        out = out + term
    return out


def default_scales(shape):
    xy = int(np.floor(np.log(max(shape[1], shape[2])) / np.log(2.0)))
    z = int(np.floor(np.log(shape[0]) / np.log(2.0)))
    return xy, z


def subbands(data, xy_scales, z_scales, kernel=B3SPLINE, mode="periodic"):
    """Yield (xy_scale, z_scale, coefficients) in the order of the original implementation."""
    coarse = np.asarray(data, dtype=np.float32)
    for xy in range(xy_scales + 1):
        if xy < xy_scales:
            sm = smooth(smooth(coarse, kernel, 2**xy, 2, mode), kernel, 2**xy, 1, mode)
            detail = coarse - sm
        else:
            detail = coarse
        c = detail
        for z in range(z_scales):
            s = smooth(c, kernel, 2**z, 0, mode)
            yield xy, z, c - s
            c = s
        yield xy, z_scales, c
        if xy < xy_scales:
            coarse = sm


def mad_threshold(w, sigma):
    a = np.sort(np.abs(w).ravel())
    n = a.size
    median = (float(a[(n - 1) // 2]) + float(a[n // 2])) / 2.0
    return np.float32(median / 0.6745 * sigma)


def denoise(
    data,
    xy_scales=-1,
    z_scales=-1,
    iterations=3,
    positivity=True,
    xy_approx=False,
    z_approx=False,
    total_power=False,
    sigma=None,
    thresholds=None,
    rule="mrs",
    fix_mrs_after=None,
    mode="periodic",
):
    """
    Iterative 2D-1D de-noising. Thresholds are adaptive (sigma times the
    median absolute deviation of each sub-band) if sigma is given, otherwise
    the fixed thresholds are used. rule is 'mrs', 'hard' or 'soft'.
    """
    data = np.asarray(data, dtype=np.float32)
    dxy, dz = default_scales(data.shape)
    XY = dxy if xy_scales < 0 else xy_scales
    Z = dz if z_scales < 0 else z_scales
    rec = np.zeros_like(data)
    mrs = {}
    if thresholds is None:
        thresholds = np.zeros((XY + 1, Z + 1), dtype=np.float32)
    thresholds = np.array(thresholds, dtype=np.float32)

    for it in range(iterations):
        fixed = fix_mrs_after is not None and it >= fix_mrs_after
        for xy, z, w in subbands(data - rec, XY, Z, mode=mode):
            if (xy < XY or xy_approx) and (z < Z or z_approx):
                if sigma is not None and not (rule == "mrs" and fixed):
                    thresholds[xy, z] = mad_threshold(w, sigma)
                t = thresholds[xy, z]
                if rule == "mrs":
                    m = mrs.setdefault((xy, z), np.zeros(data.shape, dtype=bool))
                    if not fixed and t >= 0:
                        m |= np.abs(w) > t
                    rec[m] += w[m]
                elif t >= 0:
                    sel = np.abs(w) > t
                    if rule == "hard":
                        rec[sel] += w[sel]
                    else:
                        rec[sel] += (np.sign(w) * (np.abs(w) - t))[sel]
            elif xy == XY and z == Z and total_power:
                rec += w
        if positivity:
            rec[rec < 0] = 0.0
    return rec, thresholds, mrs
