# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Cython compatibility fix for NumPy 2.0+
cimport cython
import numpy as np
cimport numpy as np

# Ensure NumPy C types are declared
ctypedef np.float32_t float32_t
ctypedef np.int32_t int32_t
ctypedef np.int64_t int64_t

cdef inline float32_t max(float32_t a, float32_t b) nogil:
    return a if a >= b else b

cdef inline float32_t min(float32_t a, float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(np.ndarray[float32_t, ndim=2] dets, float32_t thresh):
    cdef np.ndarray[float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[int64_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]

    # FIXED: use int32_t array and valid NumPy dtype
    cdef np.ndarray[int32_t, ndim=1] suppressed = np.zeros((ndets,), dtype=np.int32)

    # Loop indices
    cdef int _i, _j
    cdef int i, j

    # Box i (the one currently under consideration)
    cdef float32_t ix1, iy1, ix2, iy2, iarea

    # For overlap calculation with box j
    cdef float32_t xx1, yy1, xx2, yy2
    cdef float32_t w, h, inter, ovr

    with nogil:
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])

                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)

                if ovr >= thresh:
                    suppressed[j] = 1

    return np.where(suppressed == 0)[0]
