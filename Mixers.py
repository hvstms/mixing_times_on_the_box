import numpy as np
import warnings


# === Functions === #

# Shifting the entries of matrices by adding a 0 row/column
def shift(matrix, direction):
    n = matrix.shape[0]
    if direction == 'u':
        return np.block([[matrix[1:, :]], [np.zeros([1, n])]])
    if direction == 'd':
        return np.block([[np.zeros([1, n])], [matrix[:-1, :]]])
    if direction == 'l':
        return np.block([[matrix[:, 1:], np.zeros([n, 1])]])
    if direction == 'r':
        return np.block([[np.zeros([n, 1]), matrix[:, :-1]]])


# === Classes === #

# A single distribution on the n by n grid
class Distribution:

    def __init__(self, m):
        self.dist = m / m.sum()

    def __eq__(self, other, tolerance=1e-10):
        return np.allclose(self.dist, other.dist, rtol=tolerance, atol=tolerance)

    def update(self, tensor):
        s, u, d, l, r = tensor.tt * self.dist
        self.dist = s + shift(u, 'u') + shift(d, 'd') + shift(l, 'l') + shift(r, 'r')


# Parent for all random walks on the grid 
class TransitionTensor:

    @classmethod
    def leak(cls, tt):
        s, u, d, l, r = tt
        return u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum()

    @classmethod
    def div_free(cls, tt):
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


# Random entries
class RandomTensor(TransitionTensor):

    def __init__(self, d):
        tt = np.random.random([4, d, d])

        for i in range(d):
            tt[2, i, 0] = 0
            tt[3, i, d - 1] = 0

        for j in range(d):
            tt[0, 0, j] = 0
            tt[1, d - 1, j] = 0

        tt = np.block([[[np.ones([d, d])]], [[tt / tt.sum(0)]]])
        super().__init__(tt)

    def __str__(self):
        return 'RandomTensor'


# Lazy Random Walk
class LazyRandomWalk(TransitionTensor):

    def __init__(self, d):
        tt = np.block([[[4 * np.ones([d, d])]], [[np.ones([4, d, d])]]])

        tt[1, 0] -= 1
        tt[2, -1] -= 1
        tt[3, :, 0] -= 1
        tt[4, :, -1] -= 1

        tt[0, 0] += 1
        tt[0, -1] += 1
        tt[0, :, 0] += 1
        tt[0, :, -1] += 1

        super().__init__(tt)

    def __str__(self):
        return 'LazyRandomWalk'


# Simple Swirl: you stay in your lane with a lazy component
class SimpleSwirl(TransitionTensor):

    def __init__(self, d):
        tt = np.block([[[np.ones([d, d]) * .5]], [[np.zeros([4, d, d])]]])

        for i in range(d):
            for j in range(d):

                # === Swirl ===#
                # first quadrant
                if i >= j and i + j < d - 1:
                    tt[2, i, j] += .5

                # second quadrant
                if i > j and i + j >= d - 1:
                    tt[4, i, j] += .5

                # third quadrant
                if i <= j and i + j > d - 1:
                    tt[1, i, j] += .5

                # fourth quadrant
                if i < j and i + j < d:
                    tt[3, i, j] += .5

        super().__init__(tt)

    def __str__(self):
        return 'SimpleSwirl'


# Diffusion Swirl: you can change lanes
class DiffusionSwirl(TransitionTensor):
    
    def __init__(self, d):
        diff = 1 / d  # should it be a parameter?

        tt = np.block([[[np.ones([d, d]) * .5]], [[np.zeros([4, d, d])]]])

        for i in range(d):
            for j in range(d):

                # === Swirl ===#
                # first quadrant
                if i >= j and i + j < d - 1:
                    tt[2, i, j] += .5 - 2 * diff

                # second quadrant
                if i > j and i + j >= d - 1:
                    tt[4, i, j] += .5 - 2 * diff

                # third quadrant
                if i <= j and i + j > d - 1:
                    tt[1, i, j] += .5 - 2 * diff

                # fourth quadrant
                if i < j and i + j < d:
                    tt[3, i, j] += .5 - 2 * diff

                # === Diffusion ===#
                # vertical
                if (i > j and i + j > d - 1) or (i < j and i + j < d - 1):
                    if i != 0:
                        tt[1, i, j] += diff
                    if i != d - 1:
                        tt[2, i, j] += diff

                # horizontal
                if (i < j and i + j > d - 1) or (i > j and i + j < d - 1):
                    if j != 0:
                        tt[3, i, j] += diff
                    if j != d - 1:
                        tt[4, i, j] += diff

                # diagonal
                if i == j:
                    if 0 < i < d / 2:
                        tt[1, i, j] += diff
                        tt[3, i, j] += diff

                    if d - 1 > i >= d / 2:
                        tt[2, i, j] += diff
                        tt[4, i, j] += diff

                # off-diagonal
                if i + j == d - 1:
                    if 0 < j < d / 2:
                        tt[2, i, j] += diff
                        tt[3, i, j] += diff

                    if d - 1 > j >= d / 2:
                        tt[1, i, j] += diff
                        tt[4, i, j] += diff

                # === Rest ===#
                # corners
                if i in [0, d - 1] and j in [0, d - 1]:
                    tt[0, i, j] += diff

                # borders
                if i < d - 1 and j == 0:
                    tt[2, i, j] += diff

                if i == d - 1 and j < d - 1:
                    tt[4, i, j] += diff

                if i > 0 and j == d - 1:
                    tt[1, i, j] += diff

                if i == 0 and j > 0:
                    tt[3, i, j] += diff

        super().__init__(tt)

    def __str__(self):
        return 'DiffusionSwirl'

