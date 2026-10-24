# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Undecimated (a trous) 2D-1D wavelet transform and de-noising of spectral cubes.

All cubes are expected in (spectral, y, x) order, i.e. the first axis is the
spectral axis. Internally everything is single precision and C-contiguous.

Performance notes
-----------------
* All heavy loops run without the GIL and are parallelised with OpenMP. The
  number of threads follows ``OMP_NUM_THREADS`` or :func:`set_num_threads`.
* The de-noising classes never materialise a whole wavelet sub-band. After the
  spatial smoothing of one spatial scale, the spectral decomposition is carried
  out on small tiles of spectra that stay in cache, and all spectral sub-bands
  of a tile are thresholded right away. This needs two cube-sized work buffers
  instead of three and streams the cube through main memory a few times per
  spatial scale instead of several times per sub-band.
* Sub-band noise levels (median absolute coefficient) are computed exactly
  without sorting or copying the sub-band: a streaming pass histograms the
  leading bits of the coefficients, a second one keeps the few coefficients
  in the bin of the median, and a selection among those yields the median.
* The multi-resolution support is stored as one bit per coefficient instead of
  one ``short``.
"""

from cython.parallel cimport prange, parallel, threadid
from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport memcpy
from libc.math cimport fabsf, INFINITY

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef extern from "<algorithm>" namespace "std" nogil:
    void nth_element[T](T first, T nth, T last)
    T min_element[T](T first, T last)


# Boundary handling of the convolutions
BOUNDARY_MODES = {'periodic': 0, 'reflect': 1, 'zero': 2}

b3spline = np.array([1., 4., 6., 4., 1.], dtype=np.single) / 16.

# Tuning constants
cdef enum:
    YSTRIP = 64              # width (in pixels) of the column strips of the y convolution
    ZCHUNK = 4096            # pixels per work item of the cube-wide z convolution
    TILE_VALUES = 32768      # target number of coefficients in a spectral tile
    MIN_TILE_WIDTH = 16
    MAX_TILE_WIDTH = 256
    HIST_SHIFT = 17          # sub-band histograms are over the top 31 - HIST_SHIFT bits of |w|
    HIST_BINS = 16384        # 1 << (31 - HIST_SHIFT)

# What to do with a sub-band
cdef enum:
    ACT_NONE = 0
    ACT_THRESHOLD = 1
    ACT_TOTAL_POWER = 2

# Thresholding rules
cdef enum:
    RULE_HARD = 0
    RULE_HARD_MRS = 1
    RULE_SOFT = 2

# What the tile engine does with the coefficients
cdef enum:
    MODE_APPLY = 0
    MODE_HIST = 1
    MODE_STATS = 2


def openmp_enabled():
    """Whether the extension was built with OpenMP, i.e. can use several threads."""
    return bool(S2_HAVE_OPENMP)


def set_num_threads(int n):
    """Set the number of OpenMP threads used by all transforms."""
    omp_set_num_threads(max(1, n))


def get_num_threads():
    """Return the number of OpenMP threads used by all transforms."""
    return omp_get_max_threads()


def _index_table(Py_ssize_t size, Py_ssize_t ksize, Py_ssize_t step, int mode):
    """
    Return a (ksize, size) table of the sample positions entering the a trous
    convolution with the given step size. Positions outside the data are mapped
    according to the boundary mode; -1 means the sample is to be ignored.
    """
    offsets = (np.arange(ksize, dtype=np.intp) - ksize // 2) * step
    idx = np.arange(size, dtype=np.intp)[None, :] + offsets[:, None]
    if mode == 0:
        idx = idx % size
    elif mode == 1:
        if size == 1:
            idx[:] = 0
        else:
            period = 2 * (size - 1)
            idx = idx % period
            idx = np.where(idx >= size, period - idx, idx)
    else:
        idx[(idx < 0) | (idx >= size)] = -1
    return np.ascontiguousarray(idx, dtype=np.intp)


cdef inline float* _fptr(cnp.ndarray a):
    return <float*>cnp.PyArray_DATA(a)


cdef inline Py_ssize_t* _iptr(cnp.ndarray a):
    return <Py_ssize_t*>cnp.PyArray_DATA(a)


# ---------------------------------------------------------------------------
# Convolution kernels
#
# All of them compute sum_j kernel[j] * sample_j in single precision and in
# order of increasing j, so that the result does not depend on the code path.
# ---------------------------------------------------------------------------

cdef extern from "_kernels.h" nogil:
    int S2_HAVE_OPENMP
    int omp_get_max_threads()
    void omp_set_num_threads(int n)
    void s2_conv5(const float* r0, const float* r1, const float* r2, const float* r3, const float* r4,
                  float* out, Py_ssize_t n, const float* k)
    void s2_conv5_detail(const float* r0, const float* r1, const float* r2, const float* r3, const float* r4,
                         const float* centre, float* smooth, float* detail, Py_ssize_t n, const float* k)
    void s2_conv5_subtract(const float* r0, const float* r1, const float* r2, const float* r3, const float* r4,
                           float* smooth, float* a, Py_ssize_t n, const float* k)
    void s2_difference(const float* a, const float* b, float* detail, Py_ssize_t n)
    void s2_add(float* r, const float* w, Py_ssize_t n)
    void s2_add_hard(float* r, const float* w, Py_ssize_t n, float t)
    void s2_add_soft(float* r, const float* w, Py_ssize_t n, float t)
    void s2_add_mrs(float* r, const float* w, unsigned char* m, Py_ssize_t n, float t, int fixed)
    void s2_hist(const float* w, Py_ssize_t n, unsigned long long* hist, int shift)
    Py_ssize_t s2_collect(const float* w, Py_ssize_t n, float lo, float hi, float* out, Py_ssize_t* below)


cdef void _conv_line(const float* src, float* dst, Py_ssize_t n,
                     const float* kernel, Py_ssize_t ksize, Py_ssize_t step,
                     const Py_ssize_t* tab) noexcept nogil:
    """Convolve a contiguous line of n samples (src and dst must not overlap)."""
    cdef:
        Py_ssize_t h = ksize // 2
        Py_ssize_t lo = h * step
        Py_ssize_t hi = n - (ksize - 1 - h) * step
        Py_ssize_t i, j
        float a

    if lo > n:
        lo = n
    if hi < lo:
        hi = lo

    if ksize == 5:
        s2_conv5(src + lo - 2 * step, src + lo - step, src + lo, src + lo + step, src + lo + 2 * step,
                 dst + lo, hi - lo, kernel)
    else:
        for i in range(lo, hi):
            a = 0.
            for j in range(ksize):
                a = a + src[i + (j - h) * step] * kernel[j]
            dst[i] = a

    # Samples close to the boundaries
    for i in range(lo):
        dst[i] = _conv_point(src, n, i, kernel, ksize, tab)
    for i in range(hi, n):
        dst[i] = _conv_point(src, n, i, kernel, ksize, tab)


cdef inline float _conv_point(const float* src, Py_ssize_t n, Py_ssize_t i,
                              const float* kernel, Py_ssize_t ksize, const Py_ssize_t* tab) noexcept nogil:
    cdef:
        Py_ssize_t j, idx
        float a = 0.
    for j in range(ksize):
        idx = tab[j * n + i]
        if idx >= 0:
            a = a + src[idx] * kernel[j]
    return a


cdef inline bint _rows5(const float* src, Py_ssize_t stride, Py_ssize_t row, Py_ssize_t nrows,
                        Py_ssize_t ksize, const Py_ssize_t* tab, const float** rows) noexcept nogil:
    """The five input rows of output row `row` of a 5-tap convolution, if all of them exist."""
    cdef:
        Py_ssize_t j, q
    if ksize != 5:
        return False
    for j in range(5):
        q = tab[j * nrows + row]
        if q < 0:
            return False
        rows[j] = src + q * stride
    return True


cdef inline void _conv_rows(const float* src, Py_ssize_t stride, float* dst, Py_ssize_t length,
                            Py_ssize_t row, Py_ssize_t nrows,
                            const float* kernel, Py_ssize_t ksize,
                            const Py_ssize_t* tab) noexcept nogil:
    """
    Convolve along the slow axis of an array of nrows rows with the given
    stride: dst[:length] receives output row `row`. Vectorises along the rows.
    """
    cdef:
        Py_ssize_t i, j, q
        float kj
        const float* s
        const float* r[5]

    if _rows5(src, stride, row, nrows, ksize, tab, r):
        s2_conv5(r[0], r[1], r[2], r[3], r[4], dst, length, kernel)
        return

    for i in range(length):
        dst[i] = 0.
    for j in range(ksize):
        q = tab[j * nrows + row]
        if q < 0:
            continue
        kj = kernel[j]
        s = src + q * stride
        for i in range(length):
            dst[i] = dst[i] + s[i] * kj


cdef inline void _conv_rows_detail(const float* src, Py_ssize_t stride, float* smooth, float* detail,
                                   Py_ssize_t length, Py_ssize_t row, Py_ssize_t nrows,
                                   const float* kernel, Py_ssize_t ksize,
                                   const Py_ssize_t* tab) noexcept nogil:
    """Like _conv_rows, and also detail = src row - smooth."""
    cdef const float* r[5]
    if _rows5(src, stride, row, nrows, ksize, tab, r):
        s2_conv5_detail(r[0], r[1], r[2], r[3], r[4], src + row * stride, smooth, detail, length, kernel)
    else:
        _conv_rows(src, stride, smooth, length, row, nrows, kernel, ksize, tab)
        s2_difference(src + row * stride, smooth, detail, length)


cdef int _smooth_xy(float* A, float* B, Py_ssize_t nz, Py_ssize_t ny, Py_ssize_t nx,
                    const float* kernel, Py_ssize_t ksize, Py_ssize_t step,
                    const Py_ssize_t* xtab, const Py_ssize_t* ytab, bint detail) noexcept nogil:
    """
    B = smooth_y(smooth_x(A)) plane by plane; if detail, also A = A - B.
    The y convolution is done in place on B through narrow column strips.
    Returns 0 on success and -1 if scratch memory could not be allocated.
    """
    cdef:
        Py_ssize_t r, t, z, y, x0, w, i
        Py_ssize_t nstrips = (nx + YSTRIP - 1) // YSTRIP
        Py_ssize_t plane = ny * nx
        float* scratch
        float* acc
        float* brow
        float* arow
        const float* rows[5]
        int failed = 0
        int* fail = &failed

    scratch = NULL
    acc = NULL
    for r in prange(nz * ny, schedule='static'):
        _conv_line(A + r * nx, B + r * nx, nx, kernel, ksize, step, xtab)

    with parallel():
        scratch = <float*>malloc(ny * YSTRIP * sizeof(float))
        acc = <float*>malloc(YSTRIP * sizeof(float))
        for t in prange(nz * nstrips, schedule='static'):
            if scratch == NULL or acc == NULL:
                fail[0] = 1
                continue
            z = t // nstrips
            x0 = (t % nstrips) * YSTRIP
            w = nx - x0
            if w > YSTRIP:
                w = YSTRIP
            for y in range(ny):
                memcpy(scratch + y * w, B + z * plane + y * nx + x0, w * sizeof(float))
            for y in range(ny):
                brow = B + z * plane + y * nx + x0
                arow = A + z * plane + y * nx + x0
                if _rows5(scratch, w, y, ny, ksize, ytab, rows):
                    if detail:
                        s2_conv5_subtract(rows[0], rows[1], rows[2], rows[3], rows[4], brow, arow, w, kernel)
                    else:
                        s2_conv5(rows[0], rows[1], rows[2], rows[3], rows[4], brow, w, kernel)
                    continue
                _conv_rows(scratch, w, acc, w, y, ny, kernel, ksize, ytab)
                if detail:
                    for i in range(w):
                        brow[i] = acc[i]
                        arow[i] = arow[i] - acc[i]
                else:
                    for i in range(w):
                        brow[i] = acc[i]
        free(scratch)
        free(acc)
    return -failed


cdef void _smooth_z(const float* src, float* dst, Py_ssize_t nz, Py_ssize_t plane,
                    const float* kernel, Py_ssize_t ksize, const Py_ssize_t* ztab) noexcept nogil:
    """dst = smooth_z(src) for a whole cube."""
    cdef:
        Py_ssize_t t, z, c0, length
        Py_ssize_t nchunks = (plane + ZCHUNK - 1) // ZCHUNK

    for t in prange(nz * nchunks, schedule='static'):
        z = t // nchunks
        c0 = (t % nchunks) * ZCHUNK
        length = plane - c0
        if length > ZCHUNK:
            length = ZCHUNK
        _conv_rows(src + c0, plane, dst + z * plane + c0, length, z, nz, kernel, ksize, ztab)


cdef void _subtract(float* a, const float* b, Py_ssize_t n) noexcept nogil:
    """a = a - b"""
    cdef Py_ssize_t i
    for i in prange(n, schedule='static'):
        a[i] = a[i] - b[i]


cdef void _difference(float* out, const float* a, const float* b, Py_ssize_t n) noexcept nogil:
    """out = a - b"""
    cdef Py_ssize_t i
    for i in prange(n, schedule='static'):
        out[i] = a[i] - b[i]


cdef void _clip_negative(float* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in prange(n, schedule='static'):
        if a[i] < 0:
            a[i] = 0.


# ---------------------------------------------------------------------------
# Tile engine for the spectral part of the de-noising
# ---------------------------------------------------------------------------

ctypedef struct FVec:
    float* data
    Py_ssize_t size
    Py_ssize_t cap


cdef inline int _reserve(FVec* v, Py_ssize_t n) noexcept nogil:
    cdef:
        Py_ssize_t cap
        float* data
    if v.cap >= n:
        return 0
    cap = 2 * v.cap
    if cap < n:
        cap = n
    data = <float*>realloc(v.data, cap * sizeof(float))
    if data == NULL:
        return -1
    v.data = data
    v.cap = cap
    return 0


ctypedef struct RowCtx:
    # geometry
    Py_ssize_t nz, ny, nx, plane
    Py_ssize_t W, ncx, ntiles, tile_cap
    # spectral decomposition
    const float* kernel
    Py_ssize_t ksize
    const Py_ssize_t* ztab      # (nzscales, ksize, nz)
    int nzscales
    int zmax                    # number of spectral smoothing steps to carry out
    int mode
    # MODE_APPLY
    const int* action           # (nzscales + 1)
    const float* thr            # (nzscales + 1)
    int rule
    bint fix_mrs
    unsigned char* mrs          # MRS bits of sub-band (xy, 0); NULL if unused
    Py_ssize_t nxb              # bytes per MRS row
    Py_ssize_t mrs_zstride      # bytes per MRS sub-band
    float* rec
    # MODE_HIST and MODE_STATS
    const int* need             # (nzscales + 1)
    unsigned long long* hist    # (nthreads, nzscales + 1, HIST_BINS)
    const float* lo             # (nzscales + 1)
    const float* hi             # (nzscales + 1)
    Py_ssize_t* below           # (nthreads, nzscales + 1)
    FVec* coll                  # (nthreads, nzscales + 1)
    int* fail


cdef void _handle_row(RowCtx* c, const float* w, int zsub, Py_ssize_t z, Py_ssize_t y, Py_ssize_t x0,
                      Py_ssize_t wt, int tid) noexcept nogil:
    """Handle row z (length wt, starting at y, x0) of the coefficients of spectral sub-band zsub."""
    cdef:
        int act
        float t
        float* r
        FVec* vec
        Py_ssize_t k

    if c.mode == MODE_APPLY:
        act = c.action[zsub]
        if act == ACT_NONE:
            return
        r = c.rec + z * c.plane + y * c.nx + x0
        if act == ACT_TOTAL_POWER:
            s2_add(r, w, wt)
            return
        t = c.thr[zsub]
        if t < 0:
            t = INFINITY
        if c.rule == RULE_HARD_MRS:
            s2_add_mrs(r, w, c.mrs + zsub * c.mrs_zstride + (z * c.ny + y) * c.nxb + x0 // 8,
                       wt, t, c.fix_mrs)
        elif c.thr[zsub] >= 0:
            if c.rule == RULE_HARD:
                s2_add_hard(r, w, wt, t)
            else:
                s2_add_soft(r, w, wt, t)

    elif c.mode == MODE_HIST:
        if c.need[zsub]:
            s2_hist(w, wt, c.hist + (tid * (c.nzscales + 1) + zsub) * HIST_BINS, HIST_SHIFT)

    elif c.need[zsub]:
        k = tid * (c.nzscales + 1) + zsub
        vec = &c.coll[k]
        if _reserve(vec, vec.size + wt) < 0:
            c.fail[0] = 1
            return
        vec.size += s2_collect(w, wt, c.lo[zsub], c.hi[zsub], vec.data + vec.size, &c.below[k])


cdef void _run_tile(RowCtx* c, const float* D, Py_ssize_t tile,
                    float* T0, float* T1, float* wrow, int tid) noexcept nogil:
    """
    Spectral decomposition of one tile of spectra. Each row of each sub-band is
    handed to _handle_row as soon as it has been computed.
    """
    cdef:
        Py_ssize_t y = tile // c.ncx
        Py_ssize_t x0 = (tile % c.ncx) * c.W
        Py_ssize_t wt = c.nx - x0
        Py_ssize_t z
        int zs
        float* tmp
        const Py_ssize_t* tab

    if wt > c.W:
        wt = c.W

    for z in range(c.nz):
        memcpy(T0 + z * wt, D + z * c.plane + y * c.nx + x0, wt * sizeof(float))

    for zs in range(c.zmax):
        tab = c.ztab + zs * c.ksize * c.nz
        for z in range(c.nz):
            # Smoothed spectra into T1, wavelet coefficients into wrow
            _conv_rows_detail(T0, wt, T1 + z * wt, wrow, wt, z, c.nz, c.kernel, c.ksize, tab)
            _handle_row(c, wrow, zs, z, y, x0, wt, tid)
        tmp = T0
        T0 = T1
        T1 = tmp

    if c.zmax == c.nzscales:
        for z in range(c.nz):
            _handle_row(c, T0 + z * wt, c.nzscales, z, y, x0, wt, tid)


cdef int _run_tiles(RowCtx* c, const float* D, int nthreads) noexcept nogil:
    cdef:
        Py_ssize_t tile
        float* T0
        float* T1
        float* wrow
        int tid
        int failed = 0

    c.fail = &failed
    T0 = T1 = wrow = NULL
    tid = 0

    with parallel(num_threads=nthreads):
        tid = threadid()
        T0 = <float*>malloc(c.tile_cap * sizeof(float))
        T1 = <float*>malloc(c.tile_cap * sizeof(float))
        wrow = <float*>malloc(c.W * sizeof(float))
        for tile in prange(c.ntiles, schedule='dynamic', chunksize=4):
            if T0 == NULL or T1 == NULL or wrow == NULL:
                c.fail[0] = 1
                continue
            _run_tile(c, D, tile, T0, T1, wrow, tid)
        free(T0)
        free(T1)
        free(wrow)
    return -failed


cdef void _select_medians(FVec* merged, const Py_ssize_t* r1, const Py_ssize_t* r2,
                          float* v1, float* v2, Py_ssize_t n) noexcept nogil:
    """Exact order statistics r1 <= r2 (r2 <= r1 + 1) of each of the n value sets."""
    cdef:
        Py_ssize_t k
        float* d
    for k in prange(n, schedule='dynamic', chunksize=1):
        d = merged[k].data
        nth_element(d, d + r1[k], d + merged[k].size)
        v1[k] = d[r1[k]]
        if r2[k] == r1[k]:
            v2[k] = v1[k]
        else:
            v2[k] = min_element(d + r1[k] + 1, d + merged[k].size)[0]


# ---------------------------------------------------------------------------
# Generic 2D-1D decomposition
# ---------------------------------------------------------------------------

cdef class WaveletDecomposition2D1D:
    """
    Base class for 2D-1D wavelet decomposition after Starck et al. 2009 and Flöer & Winkel 2012.

    decompose() materialises every sub-band in the work array and hands it to
    handle_coefficients(), which can be overridden (also from Python). The
    de-noising subclasses use a much faster fused engine instead.
    """

    cdef:
        cnp.ndarray _work, _data
        int _xy_scales, _z_scales
        cnp.ndarray _xy_mother_function, _z_mother_function
        int _nwork, _boundary

    def __cinit__(self, *args, **kwargs):
        self._nwork = 3
        self._boundary = 0

    def __init__(self, data, xy_scales=-1, z_scales=-1):
        """
        Initialize the decomposition object.

        Parameters
        ----------

        data : 3D numpy array
            The data to decompose. The data are converted to float, i.e. single
            accuracy, values. If the data already are a C-contiguous float32
            array, they are referenced, not copied; they are never modified.

        xy_scales, z_scales : integer, optional
            The spatial and spectral scales to use for data decomposition.
            If either is less than 0 or not given, the scales are determined from the data, e.g. z_scales = floor(log_2(data.shape[0])).
            If either scale is 0, no decomposition is performed for the corresponding axes.
        """

        self.data = data

        if self._xy_mother_function is None:
            self.xy_mother_function = b3spline
        if self._z_mother_function is None:
            self.z_mother_function = b3spline

        self._work = np.empty((self._nwork,) + np.shape(self._data), dtype=np.single)

        self.xy_scales = xy_scales
        self.z_scales = z_scales


    property work:
        """
        ndarray containing the working space of the transform
        """
        def __get__(self):
            return self._work


    property data:
        """
        ndarray containing the original data to be reconstructed
        """
        def __get__(self):
            return self._data
        def __set__(self, value):
            value = np.ascontiguousarray(value, dtype=np.single)
            if value.ndim != 3:
                raise ValueError("Data must be three-dimensional")
            if self._data is not None and value.shape != np.shape(self._data):
                raise ValueError("Data must keep its shape")
            self._data = value


    property xy_mother_function:
        """
        ndarray containing the wavelet mother function
        for the spatial part of the transform
        """
        def __get__(self):
            return self._xy_mother_function
        def __set__(self, value):
            self._xy_mother_function = np.array(value, dtype=np.single).ravel()


    property z_mother_function:
        """
        narray containing the wavelet mother function
        for the spectral part of the transform
        """
        def __get__(self):
            return self._z_mother_function
        def __set__(self, value):
            self._z_mother_function = np.array(value, dtype=np.single).ravel()


    property xy_scales:
        """
        the number of spatial scales to use during the reconstruction
        """
        def __get__(self):
            return self._xy_scales
        def __set__(self, value):
            if value < 0:
                self._xy_scales = int(np.floor(np.log(max(self._data.shape[1], self._data.shape[2])) / np.log(2.0)))
            else:
                self._xy_scales = int(np.floor(value))


    property z_scales:
        """
        The number of spectral scales to use during the reconstruction
        """
        def __get__(self):
            return self._z_scales
        def __set__(self, value):
            if value < 0:
                self._z_scales = int(np.floor(np.log(self._data.shape[0]) / np.log(2.0)))
            else:
                self._z_scales = int(np.floor(value))


    property boundary:
        """
        Boundary handling of the convolutions: 'periodic' (default), 'reflect' or 'zero'
        """
        def __get__(self):
            return {v: k for k, v in BOUNDARY_MODES.items()}[self._boundary]
        def __set__(self, value):
            if value not in BOUNDARY_MODES:
                raise ValueError("Boundary must be one of %s" % ', '.join(BOUNDARY_MODES))
            self._boundary = BOUNDARY_MODES[value]


    def _tables(self):
        """Index tables of all convolutions."""
        cdef Py_ssize_t nz = self._data.shape[0], ny = self._data.shape[1], nx = self._data.shape[2]
        kxy = self._xy_mother_function.shape[0]
        kz = self._z_mother_function.shape[0]
        xtabs = [_index_table(nx, kxy, 1 << s, self._boundary) for s in range(self._xy_scales)]
        ytabs = [_index_table(ny, kxy, 1 << s, self._boundary) for s in range(self._xy_scales)]
        ztab = np.empty((max(self._z_scales, 1), kz, nz), dtype=np.intp)
        for s in range(self._z_scales):
            ztab[s] = _index_table(nz, kz, 1 << s, self._boundary)
        return xtabs, ytabs, ztab


    def decompose(self):
        """
        Perform the wavelet decomposition. For each combination of spatial and spectral scale,
        the coefficients are calculated and the handle_coefficients member function is called.
        """

        cdef:
            Py_ssize_t nz = self._data.shape[0], ny = self._data.shape[1], nx = self._data.shape[2]
            Py_ssize_t plane = ny * nx, size = nz * plane
            float* work = _fptr(self._work)
            float* xyk = _fptr(self._xy_mother_function)
            float* zk = _fptr(self._z_mother_function)
            Py_ssize_t kxy = self._xy_mother_function.shape[0], kz = self._z_mother_function.shape[0]
            Py_ssize_t* xtab
            Py_ssize_t* ytab
            Py_ssize_t* ztab
            int fine_xy, fine_z, coarse_xy, coarse_z, xy_scale, z_scale, rc

        xtabs, ytabs, ztabs = self._tables()
        ztab = _iptr(ztabs)

        coarse_xy = 0
        fine_xy = 1

        self.init_work()

        for xy_scale in range(self._xy_scales + 1):

            if xy_scale < self._xy_scales:
                # Smooth the coarse data into fine_xy and turn coarse_xy into the wavelet coefficients
                xtab = _iptr(xtabs[xy_scale])
                ytab = _iptr(ytabs[xy_scale])
                with nogil:
                    rc = _smooth_xy(work + coarse_xy * size, work + fine_xy * size, nz, ny, nx,
                                    xyk, kxy, 1 << xy_scale, xtab, ytab, True)
                if rc < 0:
                    raise MemoryError()

                # Exchange the indices
                coarse_xy, fine_xy = fine_xy, coarse_xy
                # Now the current coarse scale is in coarse_xy and the wavelet coefficients are in fine_xy
            else:
                # In the last iteration, the "wavelet coefficients" to be analyzed by
                # the spectral decomposition is just the maximally smoothed spatial scale
                fine_xy = coarse_xy

            # The coarse_z index points to the xy wavelet coefficients
            # and the fine_z index points to the third work array
            coarse_z = fine_xy
            fine_z = 2

            for z_scale in range(self._z_scales):

                with nogil:
                    _smooth_z(work + coarse_z * size, work + fine_z * size, nz, plane,
                              zk, kz, ztab + z_scale * kz * nz)
                    # Create the wavelet coefficients in coarse_z
                    _subtract(work + coarse_z * size, work + fine_z * size, size)

                # Exchange the indices and take care not to overwrite coarse_xy
                if z_scale % 2 == 0:
                    coarse_z = 2
                    fine_z = fine_xy
                else:
                    coarse_z = fine_xy
                    fine_z = 2
                # Now, the wavelet coefficients are in fine_z and the current coarse scale is in coarse_z

                self.handle_coefficients(fine_z, xy_scale, z_scale)

            # Handle the smooth scale of the data
            # This corresponds to the total power if xyscale == self.xy_scales
            # and to the smooth part of the spectral decomposition otherwise
            self.handle_coefficients(coarse_z, xy_scale, self._z_scales)


    cdef void init_work(self):
        """
        Work array initializer.
        """
        self._work[0] = self._data


    cpdef handle_coefficients(self, int work_array, int xy_scale, int z_scale):
        """
        To be overridden by subclasses. Standard implementation only stores the coefficients
        in the current working directory.
        """
        np.save('coefficients_{0}_{1}.npy'.format(xy_scale, z_scale), self._work[work_array])


# ---------------------------------------------------------------------------
# De-noising
# ---------------------------------------------------------------------------

cdef class Denoise2D1DHard(WaveletDecomposition2D1D):
    """
    Subclass of WaveletDecomposition2D1D which implements a hard thresholding scheme for denoising.

    Each call of decompose() decomposes the residual (data - reconstruction),
    adds the significant coefficients to the reconstruction and, if requested,
    clips the reconstruction at zero.
    """

    cdef:
        cnp.ndarray _thresholds, _reconstruction
        bint _total_power, _xy_approx, _z_approx, _positivity
        int _rule
        bint _adaptive
        double _sigma

    def __cinit__(self, *args, **kwargs):
        self._nwork = 2
        self._rule = RULE_HARD

    def __init__(self, data, total_power=False, xy_approx=False, z_approx=False, xy_scales=-1, z_scales=-1):
        """
        Parameters
        ----------
        data : 3D ndarray
            The data to be denoised.

        total_power : boolean, optional
            Should the total power (i.e. the smooth scale) be part of the reconstruction?

        xy_approx, z_approx : boolean, optional
            Should the detail-approximation-coefficients (see Starck et al. 2009) be part of the reconstruction?
        """

        self.xy_approx = xy_approx
        self.total_power = total_power
        self.z_approx = z_approx
        self.positivity = False

        super(Denoise2D1DHard, self).__init__(data=data, xy_scales=xy_scales, z_scales=z_scales)

        self._reconstruction = np.zeros(np.shape(self._data), dtype=np.single)
        self._thresholds = np.zeros((self._xy_scales + 1, self._z_scales + 1), dtype=np.single)


    property reconstruction:
        """
        ndarray containing the reconstruction of the data
        """
        def __get__(self):
            return self._reconstruction
        def __set__(self, value):
            value = np.array(value, dtype=np.single, order='C')
            if value.shape == np.shape(self._data):
                self._reconstruction = value
            else:
                raise ValueError("Reconstruction must have same shape as data")


    property thresholds:
        """
        ndarray containing the thresholds for each wavelet sub-band. Negative
        thresholds switch a sub-band off.
        """
        def __get__(self):
            return self._thresholds
        def __set__(self, value):
            value = np.array(value, dtype=np.single, order='C')
            if value.shape == (self._xy_scales + 1, self._z_scales + 1):
                self._thresholds = value
            else:
                raise ValueError(
                    "Thresholds must have shape %s"
                    % ((self._xy_scales + 1, self._z_scales + 1),))


    property total_power:
        """
        Whether to add total power to reconstruction
        """
        def __get__(self):
            return self._total_power
        def __set__(self, value):
            self._total_power = bool(value)


    property xy_approx:
        """
        Whether to use wavelet sub-bands belonging to the spatial
        approximation
        """
        def __get__(self):
            return self._xy_approx
        def __set__(self, value):
            self._xy_approx = bool(value)


    property z_approx:
        """
        Whether to use wavelet sub-bands belonging to the spectral
        approximation
        """
        def __get__(self):
            return self._z_approx
        def __set__(self, value):
            self._z_approx = bool(value)


    property positivity:
        """
        Whether to enforce a positivity constraint on the reconstruction
        """
        def __get__(self):
            return self._positivity
        def __set__(self, value):
            self._positivity = bool(value)


    cdef int _action(self, int xy_scale, int z_scale):
        if (xy_scale < self._xy_scales or self._xy_approx) and (z_scale < self._z_scales or self._z_approx):
            return ACT_THRESHOLD
        elif z_scale == self._z_scales and xy_scale == self._xy_scales and self._total_power:
            return ACT_TOTAL_POWER
        return ACT_NONE


    cdef _prepare_mrs(self):
        pass


    cdef unsigned char* _mrs_row(self, int xy_scale):
        return NULL


    cdef bint _fixed_support(self):
        return False


    def decompose(self):
        """
        Perform one iteration of the de-noising: decompose the residual and add
        the significant wavelet coefficients to the reconstruction.
        """
        cdef:
            Py_ssize_t nz = self._data.shape[0], ny = self._data.shape[1], nx = self._data.shape[2]
            Py_ssize_t plane = ny * nx, size = nz * plane
            float* A = _fptr(self._work)
            float* B = A + size
            float* tmp
            float* data = _fptr(self._data)
            float* rec = _fptr(self._reconstruction)
            float* xyk = _fptr(self._xy_mother_function)
            Py_ssize_t kxy = self._xy_mother_function.shape[0]
            Py_ssize_t* xtab
            Py_ssize_t* ytab
            int xy_scale, rc

        if self._thresholds is None or self._thresholds.shape[0] != self._xy_scales + 1 \
                or self._thresholds.shape[1] != self._z_scales + 1:
            if self._adaptive:
                self._thresholds = np.zeros((self._xy_scales + 1, self._z_scales + 1), dtype=np.single)
            else:
                raise ValueError("Thresholds must have shape %s" % ((self._xy_scales + 1, self._z_scales + 1),))
        if self._work.shape[0] < 2:
            self._work = np.empty((2,) + np.shape(self._data), dtype=np.single)
            A = _fptr(self._work)
            B = A + size
        self._prepare_mrs()

        xtabs, ytabs, ztabs = self._tables()

        # The residual is to be decomposed
        with nogil:
            _difference(A, data, rec, size)

        for xy_scale in range(self._xy_scales + 1):
            if xy_scale < self._xy_scales:
                # A: wavelet coefficients of this spatial scale, B: next coarser scale
                xtab = _iptr(xtabs[xy_scale])
                ytab = _iptr(ytabs[xy_scale])
                with nogil:
                    rc = _smooth_xy(A, B, nz, ny, nx, xyk, kxy, 1 << xy_scale, xtab, ytab, True)
                if rc < 0:
                    raise MemoryError()

            self._spectral(A, xy_scale, ztabs)

            tmp = A
            A = B
            B = tmp

        if self._positivity:
            with nogil:
                _clip_negative(rec, size)


    cdef int _spectral(self, float* D, int xy_scale, cnp.ndarray ztabs) except -1:
        """Spectral decomposition and thresholding of the spatial sub-band D."""
        cdef:
            RowCtx c
            int Z = self._z_scales
            int z, rc, last = -1
            int nthreads = omp_get_max_threads()
            cnp.ndarray action = np.zeros(Z + 1, dtype=np.intc)
            cnp.ndarray thr

        for z in range(Z + 1):
            action[z] = self._action(xy_scale, z)
            if action[z] != ACT_NONE:
                last = z
        if last < 0:
            return 0

        self._init_ctx(&c, D, ztabs)
        c.action = <int*>cnp.PyArray_DATA(action)

        if self._adaptive and not self._fixed_support():
            self._adaptive_thresholds(&c, D, xy_scale, action, nthreads)

        c.zmax = Z if last == Z else last + 1

        thr = np.ascontiguousarray(self._thresholds[xy_scale], dtype=np.single)
        c.thr = _fptr(thr)
        c.rule = self._rule
        c.fix_mrs = self._fixed_support()
        c.mrs = self._mrs_row(xy_scale)
        c.nxb = (c.nx + 7) // 8
        c.mrs_zstride = c.nz * c.ny * c.nxb
        c.rec = _fptr(self._reconstruction)
        c.mode = MODE_APPLY

        with nogil:
            rc = _run_tiles(&c, D, nthreads)
        if rc < 0:
            raise MemoryError()
        return 0


    cdef int _init_ctx(self, RowCtx* c, float* D, cnp.ndarray ztabs) except -1:
        c.nz = self._data.shape[0]
        c.ny = self._data.shape[1]
        c.nx = self._data.shape[2]
        c.plane = c.ny * c.nx
        # Tiles of W spectra; W is a multiple of 8 so that MRS bytes are not shared between tiles
        W = max(MIN_TILE_WIDTH, (TILE_VALUES // c.nz) // 8 * 8)
        W = min(W, MAX_TILE_WIDTH, (c.nx + 7) // 8 * 8)
        c.W = W
        c.ncx = (c.nx + W - 1) // W
        c.ntiles = c.ny * c.ncx
        c.tile_cap = c.nz * W
        c.kernel = _fptr(self._z_mother_function)
        c.ksize = self._z_mother_function.shape[0]
        c.ztab = _iptr(ztabs)
        c.nzscales = self._z_scales
        c.mrs = NULL
        return 0


    cdef int _adaptive_thresholds(self, RowCtx* c, float* D, int xy_scale, cnp.ndarray action, int nthreads) except -1:
        """
        Set the thresholds of all thresholded sub-bands of this spatial scale
        to sigma times the median absolute deviation (as an estimate of the rms)
        of the sub-band coefficients.

        The medians are exact: a first pass histograms the top bits of the
        absolute coefficients, which locates the bin of the median and the
        number of coefficients below it. A second pass collects the few
        coefficients falling into that bin, and a selection among them
        yields the median.
        """
        cdef:
            int Z = self._z_scales
            int nsub = Z + 1
            Py_ssize_t N = c.nz * c.ny * c.nx
            Py_ssize_t k1 = (N - 1) // 2, k2 = N // 2
            Py_ssize_t i, t, total, p
            int z, j, rc, n_sel
            cnp.ndarray need = np.zeros(nsub, dtype=np.intc)
            cnp.ndarray lo = np.zeros(nsub, dtype=np.single)
            cnp.ndarray hi = np.zeros(nsub, dtype=np.single)
            cnp.ndarray hist, below, r1, r2, v1, v2
            FVec* coll = NULL
            FVec* merged = NULL
            Py_ssize_t* pr1
            Py_ssize_t* pr2
            float* pv1
            float* pv2

        zs = [z for z in range(nsub) if action[z] == ACT_THRESHOLD]
        if not zs:
            return 0
        for z in zs:
            need[z] = 1
        c.need = <int*>cnp.PyArray_DATA(need)
        c.zmax = Z if max(zs) == Z else max(zs) + 1

        # 1. Histograms of the absolute coefficients
        hist = np.zeros((nthreads, nsub, HIST_BINS), dtype=np.uint64)
        c.mode = MODE_HIST
        c.hist = <unsigned long long*>cnp.PyArray_DATA(hist)
        with nogil:
            rc = _run_tiles(c, D, nthreads)
        if rc < 0:
            raise MemoryError()
        cumulative = np.cumsum(hist.sum(axis=0, dtype=np.int64), axis=1)
        hist = None
        n_below = {}
        for z in zs:
            b1 = int(np.searchsorted(cumulative[z], k1, side='right'))
            b2 = int(np.searchsorted(cumulative[z], k2, side='right'))
            n_below[z] = int(cumulative[z, b1 - 1]) if b1 > 0 else 0
            lo[z] = np.array([b1 << HIST_SHIFT], dtype=np.uint32).view(np.single)[0]
            if b2 >= (0x7f800000 >> HIST_SHIFT):
                hi[z] = INFINITY
            else:
                hi[z] = np.array([((b2 + 1) << HIST_SHIFT) - 1], dtype=np.uint32).view(np.single)[0]

        # 2. Collect the coefficients in the bins of the median
        below = np.zeros((nthreads, nsub), dtype=np.intp)
        coll = <FVec*>calloc(nthreads * nsub, sizeof(FVec))
        merged = <FVec*>calloc(nsub, sizeof(FVec))
        try:
            if coll == NULL or merged == NULL:
                raise MemoryError()
            c.mode = MODE_STATS
            c.lo = _fptr(lo)
            c.hi = _fptr(hi)
            c.below = _iptr(below)
            c.coll = coll
            with nogil:
                rc = _run_tiles(c, D, nthreads)
            if rc < 0:
                raise MemoryError()

            # 3. Select the order statistics among the collected values
            n_sel = len(zs)
            r1 = np.empty(n_sel, dtype=np.intp)
            r2 = np.empty(n_sel, dtype=np.intp)
            v1 = np.empty(n_sel, dtype=np.single)
            v2 = np.empty(n_sel, dtype=np.single)
            counted_below = below.sum(axis=0)
            for j, z in enumerate(zs):
                total = 0
                for t in range(nthreads):
                    total += coll[t * nsub + z].size
                if counted_below[z] != n_below[z] or k2 - n_below[z] >= total:
                    raise RuntimeError("Inconsistent sub-band statistics (NaN coefficients?)")
                merged[j].data = <float*>malloc(total * sizeof(float))
                if merged[j].data == NULL:
                    raise MemoryError()
                merged[j].size = total
                p = 0
                for t in range(nthreads):
                    memcpy(merged[j].data + p, coll[t * nsub + z].data,
                           coll[t * nsub + z].size * sizeof(float))
                    p += coll[t * nsub + z].size
                r1[j] = k1 - n_below[z]
                r2[j] = k2 - n_below[z]
            pr1 = _iptr(r1)
            pr2 = _iptr(r2)
            pv1 = _fptr(v1)
            pv2 = _fptr(v2)
            with nogil:
                _select_medians(merged, pr1, pr2, pv1, pv2, n_sel)
        finally:
            if coll != NULL:
                for i in range(nthreads * nsub):
                    free(coll[i].data)
            if merged != NULL:
                for i in range(nsub):
                    free(merged[i].data)
            free(coll)
            free(merged)

        for j, z in enumerate(zs):
            median = (<double>v1[j] + <double>v2[j]) / 2.
            rms = median / 0.6745
            self._thresholds[xy_scale, z] = rms * self._sigma
        return 0


cdef class Denoise2D1DHardMRS(Denoise2D1DHard):
    """
    Subclass of Denoise2D1DHard which adds the use of a multi-resolution support
    to the denoising process.
    """

    cdef:
        cnp.ndarray _mrs
        bint _fix_mrs

    def __cinit__(self, *args, **kwargs):
        self._rule = RULE_HARD_MRS

    def __init__(self, *args, **kwargs):

        super(Denoise2D1DHardMRS, self).__init__(*args, **kwargs)
        self._fix_mrs = False
        self._prepare_mrs()


    cdef tuple _mrs_shape(self):
        return (self._xy_scales + self._xy_approx, self._z_scales + self._z_approx,
                self._data.shape[0], self._data.shape[1], (self._data.shape[2] + 7) // 8)


    cdef _prepare_mrs(self):
        shape = self._mrs_shape()
        if self._mrs is None or np.shape(self._mrs) != shape:
            self._mrs = np.zeros(shape, dtype=np.uint8)


    cdef unsigned char* _mrs_row(self, int xy_scale):
        return <unsigned char*>cnp.PyArray_DATA(self._mrs) + \
            xy_scale * self._mrs.strides[0]


    cdef bint _fixed_support(self):
        return self._fix_mrs


    property mrs:
        """
        ndarray containing the multi-resolution support with shape
        (spatial sub-bands, spectral sub-bands) + data.shape. Internally the
        support is stored as one bit per coefficient (see packed_mrs).
        """
        def __get__(self):
            self._prepare_mrs()
            return np.unpackbits(self._mrs, axis=-1, count=self._data.shape[2], bitorder='little').astype(np.short)
        def __set__(self, value):
            self._prepare_mrs()
            value = np.asarray(value)
            if value.shape == np.shape(self._mrs)[:4] + (self._data.shape[2],):
                self._mrs = np.ascontiguousarray(np.packbits(value != 0, axis=-1, bitorder='little'))


    property packed_mrs:
        """
        The multi-resolution support packed along the last axis (little bit order).
        """
        def __get__(self):
            self._prepare_mrs()
            return self._mrs


    property fix_mrs:
        """
        Whether to keep the MRS fixed during iterative reconstruction.
        """
        def __get__(self):
            return self._fix_mrs
        def __set__(self, value):
            self._fix_mrs = bool(value)


cdef class Denoise2D1DSoft(Denoise2D1DHard):
    """
    Subclass of Denoise2D1DHard which overrides the thresholding scheme to use
    the soft thresholding rule.
    """

    def __cinit__(self, *args, **kwargs):
        self._rule = RULE_SOFT


cdef class Denoise2D1DAdaptiveMRS(Denoise2D1DHardMRS):
    """
    De-noise a three-dimensional data cube using the 2D1D wavelet
    transform with a multi-resolution support as discussed in
    Starck et al. 2011 and Flöer & Winkel 2012.

    The threshold of each sub-band is sigma times its noise level, which is
    estimated from the median absolute coefficient of the sub-band in each
    iteration.

    The transform assumes that the first axis is the spectral axis.
    """

    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        self._adaptive = True
        self.xy_mother_function = b3spline
        self.z_mother_function = b3spline
        super(Denoise2D1DAdaptiveMRS, self).__init__(*args, **kwargs)


    property sigma:
        """
        The reconstruction threshold in units of the sub-band noise level
        """
        def __get__(self):
            return self._sigma
        def __set__(self, value):
            self._sigma = float(value)


# ---------------------------------------------------------------------------
# 1D and 3D decompositions
# ---------------------------------------------------------------------------

cdef void convolve_with_stepsize(float[:] source, float[:] target, float[:] kernel, int stepsize = 1) noexcept:

    cdef:
        Py_ssize_t size = source.shape[0]
        Py_ssize_t kernelsize = kernel.shape[0]
        Py_ssize_t index, i, j, k
        double a

    for i in range(size):
        index = i - kernelsize // 2 * stepsize
        a = 0.
        for j in range(kernelsize):
            k = index % size
            if k < 0:
                k += size
            a += <double>source[k] * <double>kernel[j]
            index += stepsize
        target[i] = <float>a


cdef class WaveletDecomposition1D:

    cdef:
        cnp.ndarray _work, _data
        int _scales
        cnp.ndarray _mother_function

    def __init__(self, data, scales = -1):

        self._data = np.array(data, dtype=np.single)
        self._work = np.empty((2,) + np.shape(self._data), dtype=np.single)
        self._mother_function = b3spline.copy()

        if scales < 0:
            self._scales = int(np.ceil(np.log2(self._data.shape[0])))
        else:
            self._scales = int(scales)

    property work:
        def __get__(self):
            return self._work

    property scales:
        def __get__(self):
            return self._scales

    property data:
        def __get__(self):
            return self._data

    property mother_function:
        def __get__(self):
            return self._mother_function
        def __set__(self, value):
            self._mother_function = np.array(value, dtype=np.single).ravel()

    cdef void init_work(self):

        self._work[0] = self._data

    def decompose(self):

        cdef:
            float[:,:] work = self._work
            Py_ssize_t i
            int read_from, write_to, scale_factor, scale

        self.init_work()
        scale_factor = 1

        read_from = 0
        write_to = 1

        for scale in range(self._scales):

            ### Write smoothed version to write_to
            convolve_with_stepsize(work[read_from], work[write_to], self._mother_function, scale_factor)

            ### Subtract smoothed version from original data and store the coefficients in the original data array
            for i in range(work.shape[1]):
                work[read_from, i] = work[read_from, i] - work[write_to, i]

            ### Switch indices. The smoothed data is now in read_from for the next iteration
            read_from = (read_from + 1) % 2
            write_to = (write_to + 1) % 2

            self.handle_coefficients(write_to, scale)

            scale_factor *= 2

        self.handle_coefficients(read_from, self._scales)

    cpdef handle_coefficients(self, int work_array, int scale):
        np.save('coefficients_{0}.npy'.format(scale), self._work[work_array])


cdef class Denoise1DHardMRS(WaveletDecomposition1D):

    cdef:
        cnp.ndarray _mrs
        cnp.ndarray _reconstruction
        cnp.ndarray _thresholds
        bint _fix_mrs, _total_power

    def __init__(self, data, thresholds, scales=-1, total_power=False):

        super(Denoise1DHardMRS, self).__init__(data, scales)

        self._fix_mrs = False
        self._total_power = total_power

        self._mrs = np.zeros((self.scales,) + np.shape(self._data), dtype=np.short)
        self._reconstruction = np.zeros(np.shape(self._data), dtype=np.single)
        self._thresholds = np.array(thresholds, dtype=np.single)

    property reconstruction:
        def __get__(self):
            return self._reconstruction
        def __set__(self, value):
            self._reconstruction = np.array(value, dtype=np.single)

    property thresholds:
        def __get__(self):
            return self._thresholds
        def __set__(self, value):
            self._thresholds = np.array(value, dtype=np.single)

    property total_power:
        def __get__(self):
            return self._total_power
        def __set__(self, value):
            self._total_power = bool(value)

    property fix_mrs:
        def __get__(self):
            return self._fix_mrs
        def __set__(self, value):
            self._fix_mrs = bool(value)

    property mrs:
        def __get__(self):
            return self._mrs

    cdef void init_work(self):

        cdef:
            Py_ssize_t i
            float[:,:] work = self._work
            float[:] data = self._data
            float[:] reconstruction = self._reconstruction

        for i in range(data.shape[0]):
            work[0,i] = data[i] - reconstruction[i]

    cpdef handle_coefficients(self, int work_array, int scale):

        cdef:
            Py_ssize_t i
            float[:,:] work = self._work
            float[:] reconstruction = self._reconstruction
            float[:] thresholds = self._thresholds
            short[:,:] mrs = self._mrs

        if (scale == self._scales and self._total_power):
            for i in range(reconstruction.shape[0]):
                reconstruction[i] += work[work_array, i]

        elif (scale < self._scales):
            for i in range(reconstruction.shape[0]):
                if self._fix_mrs:
                    if mrs[scale,i]:
                        reconstruction[i] += work[work_array, i]

                elif mrs[scale,i] or (fabsf(work[work_array, i]) > thresholds[scale]):
                    reconstruction[i] += work[work_array, i]
                    mrs[scale,i] = 1


cdef class WaveletDecomposition3D:

    cdef:
        cnp.ndarray _work, _data
        int _scales
        cnp.ndarray _mother_function

    def __init__(self, data, scales = -1):

        self._data = np.ascontiguousarray(data, dtype=np.single)
        if self._data.ndim != 3:
            raise ValueError("Data must be three-dimensional")
        self._work = np.empty((3,) + np.shape(self._data), dtype=np.single)
        self._mother_function = b3spline.copy()

        if scales < 0:
            self._scales = int(np.ceil(np.log2(np.amin(np.shape(self._data)))))
        else:
            self._scales = int(scales)

    property work:
        def __get__(self):
            return self._work

    property scales:
        def __get__(self):
            return self._scales

    property data:
        def __get__(self):
            return self._data

    property mother_function:
        def __get__(self):
            return self._mother_function
        def __set__(self, value):
            self._mother_function = np.array(value, dtype=np.single).ravel()

    cdef void init_work(self):

        self._work[0] = self._data


    def decompose(self):

        cdef:
            Py_ssize_t nz = self._data.shape[0], ny = self._data.shape[1], nx = self._data.shape[2]
            Py_ssize_t plane = ny * nx, size = nz * plane
            float* work = _fptr(self._work)
            float* k = _fptr(self._mother_function)
            Py_ssize_t ksize = self._mother_function.shape[0]
            Py_ssize_t* xtab
            Py_ssize_t* ytab
            Py_ssize_t* ztab
            int read_from, write_to, temp, scale_factor, scale, rc

        self.init_work()
        scale_factor = 1

        read_from = 0
        write_to = 1
        temp = 2

        for scale in range(self._scales):

            xt = _index_table(nx, ksize, scale_factor, 0)
            yt = _index_table(ny, ksize, scale_factor, 0)
            zt = _index_table(nz, ksize, scale_factor, 0)
            xtab = _iptr(xt)
            ytab = _iptr(yt)
            ztab = _iptr(zt)

            with nogil:
                ### Write smoothed version to temp and avoid overwriting read_from
                rc = _smooth_xy(work + read_from * size, work + write_to * size, nz, ny, nx,
                                k, ksize, scale_factor, xtab, ytab, False)
                if rc == 0:
                    _smooth_z(work + write_to * size, work + temp * size, nz, plane, k, ksize, ztab)

                    ### Subtract smoothed version from original data and store the coefficients in the original data array
                    _subtract(work + read_from * size, work + temp * size, size)
            if rc < 0:
                raise MemoryError()

            ### Rotate indices. The smoothed data is now in read_from for the next iteration
            read_from, write_to, temp = temp, read_from, write_to

            self.handle_coefficients(write_to, scale)

            scale_factor *= 2

        self.handle_coefficients(read_from, self._scales)

    cpdef handle_coefficients(self, int work_array, int scale):
        np.save('coefficients_{0}.npy'.format(scale), self._work[work_array])
