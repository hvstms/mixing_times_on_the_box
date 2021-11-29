import numpy as np
import warnings


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
        self.dist = m / m.sum()

    def __eq__(self, other, tolerance=1e-10):
        return np.allclose(self.dist, other.dist, rtol=tolerance, atol=tolerance)

    def update(self, tensor):
        s, u, d, l, r = tensor.tt * self.dist
        self.dist = s + shift(u, 'u') + shift(d, 'd') + shift(l, 'l') + shift(r, 'r')


class TransitionTensor:  # Parent for all random mixing strategies on the grid

    @classmethod
    def leak(cls, tt):  # gives back the total leakage
        s, u, d, l, r = tt
        return u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum()

    @classmethod
    def div_free(cls, tt):  # testing if the tensor is divergence free
        dim = tt.shape
        in_flow = tt[0] + sum([shift(i, j) for i, j in zip(tt[1:], ['u', 'd', 'l', 'r'])])

        return np.allclose(in_flow, np.ones(dim), rtol=1e-10, atol=1e-10)

    def __init__(self, tt):
        self.tt = tt
        self.tt /= tt.sum(0)

        if self.leak(self.tt) > 0:
            raise ValueError('The transition matrix has leaks')
        if not self.div_free(self.tt):
            warnings.warn('Your transition tensor in not divergence free', Warning)


class RandomTensor(TransitionTensor):  # iid environment

    def __init__(self, n):
        tt = np.random.random([4, n, n])

        # removing leaks
        tt[0, 0, :] = 0
        tt[1, n - 1, :] = 0
        tt[2, :, 0] = 0
        tt[3, :, n - 1] = 0

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
        tt[4, :, n - 1] = 0

        super().__init__(tt)

    def __str__(self):
        return self.type


class LazyRandomWalk(TransitionTensor):

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


class Swirl(TransitionTensor):

    def __init__(self, n, flow_rev=0, diff=0, t='Swirl'):
        self.type = t

        flow = .5 - flow_rev - 2 * diff

        # === laziness === #
        stay = .5 * np.ones([n, n])
        stay[[0, 0, n - 1, n - 1], [0, n - 1, 0, n - 1]] += diff
        if n % 2 == 1:
            stay[n // 2, n // 2] += .5 - 4 * diff

        # === upstream === #
        up = np.zeros([n - 1, n - 2])
        for i in range((n - 1) // 2):
            for j in range(n - 2):
                if i > n - 3 - j:
                    up[i, j] = flow
                elif i <= j < n - 2 - i:
                    up[i, j] = diff
                else:
                    up[i, j] = flow_rev

        up += np.flipud(up)
        if n % 2 == 0:
            up[n // 2 - 1, n // 2 - 1:] = flow
            up[n // 2 - 1, :n // 2 - 1] = flow_rev
        up = np.block([[flow_rev * np.ones([n - 1, 1]), up, (flow + diff) * np.ones([n - 1, 1])]])
        up = np.block([[np.zeros(n)], [up]])

        # === left-, right- and downstream === #
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # === ensemble === #
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return self.type
