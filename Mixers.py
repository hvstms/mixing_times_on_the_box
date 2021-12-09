from syslog import syslog

import numpy as np
import warnings
# from abc import ABC, abstractmethod


# ================= #
# === Functions === #
# ================= #


def shift(matrix, direction):  # Shifting the entries of matrices by concatenating a 0 row/column
    n = matrix.shape[0]
    if direction == 'u':
        return np.block([[matrix[1:, :]], [np.zeros([1, n])]])
    if direction == 'd':
        return np.block([[np.zeros([1, n])], [matrix[:-1, :]]])
    if direction == 'l':
        return np.block([[matrix[:, 1:], np.zeros([n, 1])]])
    if direction == 'r':
        return np.block([[np.zeros([n, 1]), matrix[:, :-1]]])


# =============== #
# === Classes === #
# =============== #

class Distribution:  # A single distribution on the n by n grid

    def __init__(self, m):
        self.d = m / m.sum()

    def __eq__(self, other, tolerance=1e-10):
        return np.allclose(self.d, other.d, rtol=tolerance, atol=tolerance)

    def __sub__(self, other):
        return Distribution(self.d - other.d)

    def update(self, tensor):
        s, u, d, l, r = tensor.tt * self.d
        self.d = s + shift(u, 'u') + shift(d, 'd') + shift(l, 'l') + shift(r, 'r')


class DegenerateDistribution(Distribution):
    def __init__(self, n, i, j):
        d = np.zeros([n, n])
        d[i, j] = 1

        super().__init__(d)


class TransitionTensor:  # Parent for all random mixing strategies on the grid

    @classmethod
    def leak(cls, tt):  # gives back the total leakage
        s, u, d, l, r = tt
        return u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum()

    @classmethod
    def div_free(cls, tt):  # testing if the tensor is divergence free
        dim = tt.shape[1:]
        in_flow = tt[0] + sum([shift(i, j) for i, j in zip(tt[1:], ['u', 'd', 'l', 'r'])])

        syslog('asd')

        return np.allclose(in_flow, np.ones(dim), rtol=1e-10, atol=1e-10)

    def __init__(self, tt):
        if self.leak(tt) > 0:
            raise ValueError('The transition matrix has leaks')
        if not self.div_free(tt):
            warnings.warn('Your transition tensor in not divergence free', Warning)

        self.tt = tt
        self.tt /= tt.sum(0)

    """
    @abstractmethod
    def stationary_dist(self):
        pass
    """


class SymmetricTensor(TransitionTensor):
    def __init__(self, tt):
        super().__init__(tt)

    def stationary_dist(self):
        return Distribution(np.ones(self.tt.shape))

    def distance_from_stationarity(self, dist):
        return np.abs(self.stationary_dist() - dist.d).sum() * .5


class RandomEnvironment(TransitionTensor):  # iid environment

    def __init__(self, n):
        tt = np.random.random([4, n, n])

        # removing leaks
        tt[0, 0, :] = 0
        tt[1, -1, :] = 0
        tt[2, :, 0] = 0
        tt[3, :, -1] = 0

        # creating the tensor with 1/2 laziness
        tt = np.block([[[np.ones([n, n])]], [[tt / tt.sum(0)]]])
        super().__init__(tt)

    def __str__(self):
        return 'RandomTensor'


class UpwardDrift(TransitionTensor):  # iid environment

    def __init__(self, n, diff=0, t='UpwardDrift'):
        self.type = t

        flow = .5 - diff  # amount of the flow on the edge
        tt = np.block([[[.5 * np.ones([n, n])]],
                       [[flow * np.ones([n, n])]],
                       [[np.zeros([n, n])]],
                       [[diff * np.ones([n, n])]],
                       [[diff * np.ones([n, n])]]])

        tt[1, :, 1:-1] -= diff
        tt[0, 0, :] += flow
        tt[0, 0, 1:-1] -= diff

        # removing leaks
        tt[1, 0, :] = 0
        tt[3, :, 0] = 0
        tt[4, :, -1] = 0

        super().__init__(tt)

    def __str__(self):
        return self.type


class LazyRandomWalk(SymmetricTensor):

    def __init__(self, n):
        tt = np.block([[[4 * np.ones([n, n])]], [[np.ones([4, n, n])]]])

        # removing leaks
        tt[1, 0] -= 1
        tt[2, -1] -= 1
        tt[3, :, 0] -= 1
        tt[4, :, -1] -= 1

        # these leaks are transferred to laziness
        tt[0, 0] += 1
        tt[0, -1] += 1
        tt[0, :, 0] += 1
        tt[0, :, -1] += 1

        super().__init__(tt)

    def __str__(self):
        return 'LazyRandomWalk'


class Swirl(SymmetricTensor):

    def __init__(self, depth=1, flow_rev=0, diff=0, alternating=True):
        self.type = ['', 'Alternating'][alternating] + 'Swirl' + ['', 'WithDiffusion'][diff != 0]

        flow = .5 - flow_rev - 2 * diff

        # === laziness === #
        stay = .5 * np.ones([2 * depth] * 2)
        stay[[0, 0, -1, -1], [0, -1, 0, -1]] += diff

        # === upstream === #
        def recursion(d):
            if d == 0:
                return np.empty([0, 0])

            prev = recursion(d - 1)
            n = prev.shape[0]

            curr = np.block([[prev], [diff * np.ones([1, n])]])
            curr = np.block([[diff * np.ones([1, n + 2])], [np.zeros([n + 1, 1]), curr, np.zeros([n + 1, 1])]])
            i = [0, -1][d % 2 != 1 and alternating]
            curr[1:, i] = flow

            return curr

        up = recursion(depth)
        up[0, :] = 0
        idx = [0, -1][depth % 2 != 1 and alternating]
        up[1:, idx] += diff

        # === left-, right- and downstream === #
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # === ensemble === #
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return self.type


class BalazsFlow(SymmetricTensor):

    def __init__(self, depth=1, diff=0, alternating=True):
        self.type = ['', 'Alternating'][alternating] + 'BalazsFlow' + ['', 'WithDiffusion'][diff != 0]

        flow = .5 - 2 * diff

        # === laziness === #
        stay = .5 * np.ones([2 ** (depth + 1) - 2] * 2)
        stay[[0, 0, -1, -1], [0, -1, 0, -1]] += diff

        # === upstream === #
        def recursion(d):
            if d == 0:
                return np.empty([0, 0])

            prev = recursion(d - 1)
            n = 2 * prev.shape[0]

            curr = np.block([[prev, prev], [prev, prev], [diff * np.ones([1, n])]])
            curr = np.block([[diff * np.ones([1, n + 2])], [np.zeros([n + 1, 1]), curr, np.zeros([n + 1, 1])]])
            i = [0, -1][d % 2 != 1 and alternating]
            curr[1:, i] = flow

            return curr

        up = recursion(depth)
        up[0, :] = 0
        idx = [0, -1][depth % 2 != 1 and alternating]
        up[1:, idx] += diff

        # === left-, right- and downstream === #
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # === ensemble === #
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return self.type


class TamasFlow(SymmetricTensor):

    def __init__(self, tt, alternating=True):
        super().__init__(tt)
