import math
import socket
from threading import Lock


try:
    import ray
except ModuleNotFoundError:
    ray = None
else:

    @ray.remote
    class RayRemoteLock:
        def __init__(self):
            self.lock = Lock()

        def acquire(self, block=True, timeout=-1):
            return self.lock.acquire(block, timeout)

        def release(self):
            self.lock.release()


def get_host_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def get_cluster_host_set(resource):
    return {node['NodeManagerAddress'] for node in resource if node['Alive']}


class CumulativeMovingStats:
    def __init__(self):
        self.n = 0
        self.m = 0.0
        self.s = 0.0

    def clear(self):
        self.__init__()

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            m = self.m + (x - self.m) / self.n
            s = self.s + (x - self.m) * (x - m)

            self.m = m
            self.s = s

    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


class ExponentialMovingStats:
    def __init__(self, momentum):
        assert 0.0 <= momentum < 1.0

        self.momentum = momentum
        self.n = 0
        self.m = 0.0
        self.s = 0.0

    def clear(self):
        self.__init__(self.momentum)

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            delta = x - self.m
            m = self.m + self.momentum * delta
            s = (1.0 - self.momentum) * (self.s + self.momentum * delta * delta)

            self.m = m
            self.s = s

    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s if self.n >= 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


SHARED_POLICY_ID = 'shared_policy'


def shared_policy_mapping_fn(agent_id, **kwargs):
    return SHARED_POLICY_ID
