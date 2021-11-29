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

    def __init__(self, d):
        tt = np.random.random([4, d, d])

        # removing leaks
        tt[0, 0, :] = 0
        tt[1, d - 1, :] = 0
        tt[2, :, 0] = 0
        tt[3, :, d - 1] = 0

        # creating the tensor with 1/2 laziness
        tt = np.block([[[np.ones([d, d])]], [[tt / tt.sum(0)]]])
        super().__init__(tt)

    def __str__(self):
        return 'RandomTensor'


class LazyRandomWalk(TransitionTensor):

    def __init__(self, d):
        tt = np.block([[[4 * np.ones([d, d])]], [[np.ones([4, d, d])]]])

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


class SimpleSwirl(TransitionTensor):  # distance from center is constant and there is a lazy component

    def __init__(self, n):
        up = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if j >= i > n - j - 1:
                    up[i, j] += .5

        # the rest comes from rotational symmetry
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # ensemble
        tt = np.block([[[.5 * np.ones([n, n])]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return 'SimpleSwirl'


class DiffusionSwirl(TransitionTensor):  # same as the SimpleSwirl with diffusion to the side

    def __init__(self, n):
        diff = 1 / n  # should it be a parameter?

        # laziness
        stay = .5 * np.ones([n, n])
        stay[[0, 0, n - 1, n - 1], [0, n - 1, 0, n - 1]] += diff
        if n % 2 == 1:
            stay[n // 2, n // 2] += .5 - 4 * diff

        # upstream
        up = np.zeros([n - 1, n - 2])
        for i in range((n - 1) // 2):
            for j in range(n - 2):
                if i > n - 3 - j:
                    up[i, j] = .5 - 2 * diff
                elif i <= j < n - 2 - i:
                    up[i, j] = diff

        up += np.flipud(up)
        if n % 2 == 0:
            up[n // 2 - 1, n // 2 - 1:] = .5 - 2 * diff
        up = np.block([[np.zeros([n - 1, 1]), up, (.5 - diff) * np.ones([n - 1, 1])]])
        up = np.block([[np.zeros(n)], [up]])

        # the rest comes from rotational symmetry
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # ensemble
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return 'DiffusionSwirl'


class DiffusionSwirl(TransitionTensor):  # same as the SimpleSwirl with diffusion to the side

    def __init__(self, n, ):
        diff = 1 / n  # should it be a parameter?

        # laziness
        stay = .5 * np.ones([n, n])
        stay[[0, 0, n - 1, n - 1], [0, n - 1, 0, n - 1]] += diff
        if n % 2 == 1:
            stay[n // 2, n // 2] += .5 - 4 * diff

        # upstream
        up = np.zeros([n - 1, n - 2])
        for i in range((n - 1) // 2):
            for j in range(n - 2):
                if i > n - 3 - j:
                    up[i, j] = .5 - 2 * diff
                elif i <= j < n - 2 - i:
                    up[i, j] = diff

        up += np.flipud(up)
        if n % 2 == 0:
            up[n // 2 - 1, n // 2 - 1:] = .5 - 2 * diff
        up = np.block([[np.zeros([n - 1, 1]), up, (.5 - diff) * np.ones([n - 1, 1])]])
        up = np.block([[np.zeros(n)], [up]])

        # the rest comes from rotational symmetry
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # ensemble
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        super().__init__(tt)

    def __str__(self):
        return 'DiffusionSwirl'
