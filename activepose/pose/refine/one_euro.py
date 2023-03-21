import math

import numpy as np


# ----------------------------------------------------------------------------


class LowPassFilter:
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        if isinstance(alpha, np.ndarray):
            alpha = alpha.astype(np.float32)
            if np.any(alpha <= 0) or np.any(alpha > 1.0):
                raise ValueError(f'alpha exceeds range {alpha}')
        else:
            alpha = float(alpha)
            if alpha <= 0 or alpha > 1.0:
                raise ValueError('alpha (%s) should be in (0.0, 1.0]' % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


# ----------------------------------------------------------------------------


class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError('freq should be >0')
        if mincutoff <= 0:
            raise ValueError('mincutoff should be >0')
        if dcutoff <= 0:
            raise ValueError('dcutoff should be >0')
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * np.abs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


# ----------------------------------------------------------------------------

# if __name__=="__main__":

#     import random

#     duration = 10.0 # seconds

#     config = {
#         'freq': 120,       # Hz
#         'mincutoff': 1.0,  # FIXME
#         'beta': 1.0,       # FIXME
#         'dcutoff': 1.0     # this one should be ok
#         }

#     print "#SRC OneEuroFilter.py"
#     print "#CFG %s"%config
#     print "#LOG timestamp, signal, noisy, filtered"

#     f = OneEuroFilter(**config)
#     timestamp = 0.0 # seconds
#     while timestamp < duration:
#         signal = math.sin(timestamp)
#         noisy = signal + (random.random()-0.5)/5.0
#         filtered = f(noisy, timestamp)
#         print "{0}, {1}, {2}, {3}".format(timestamp, signal, noisy, filtered)
#         timestamp += 1.0/config["freq"]
