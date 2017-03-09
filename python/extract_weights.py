#!/usr/bin/python2

import caffe
import math
import numpy as np
from sklearn.cluster import KMeans


def calcBD(layer, K = 32, M = 4, fc = False):
    D = []
    B = []
    for i in range(layer.shape[1] // M):
        s = layer[ : , i * M : (i + 1) * M]
        kmeans = KMeans(n_clusters=K, verbose=0, n_jobs=-1).fit(s)
        b = kmeans.labels_
        # TODO: fc should also work without transpose
        cluster_centers = kmeans.cluster_centers_.transpose() if fc else kmeans.cluster_centers_
        if (len(D) != 0):
            D = np.vstack((D, cluster_centers))
        else:
            D = cluster_centers
        if (len(B) != 0):
            B = np.hstack((B, b))
        else:
            B = b
    return (D, B)


def codeB(B, K = 32):
    TOTAL_BITS = 32
    BITS = int(math.log(K, 2))
    REST_BITS = TOTAL_BITS - BITS

    Bs = np.zeros((len(B) * BITS // TOTAL_BITS + np.sign(len(B) * BITS % TOTAL_BITS)), dtype=np.int32)
    total_bit_shift = 0
    for b in B:
        byte_shift = total_bit_shift // TOTAL_BITS
        bit_shift = total_bit_shift % TOTAL_BITS
        shift = REST_BITS - bit_shift
        if shift < 0:
            Bs[byte_shift] = np.bitwise_or(Bs[byte_shift], np.right_shift(b, -shift))
            Bs[byte_shift + 1] = np.bitwise_or(Bs[byte_shift + 1], np.left_shift(b, shift + TOTAL_BITS))
        else:
            Bs[byte_shift] = np.bitwise_or(Bs[byte_shift], np.left_shift(b, shift))
        total_bit_shift += BITS
    Bs.dtype = np.float32
    return Bs
