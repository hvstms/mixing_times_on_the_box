import warnings
# from syslog import syslog

from abc import abstractmethod

import numpy
import numpy as np

import matplotlib.pyplot as plt


# ================= #
# === Functions === #
# ================= #

def shift(matrix, direction):  # Shifting the entries of matrices by concatenating a 0 row/column
    n = matrix.shape[0]
    if direction == 'u':
        return np.block([[matrix[1:, :]], [np.zeros([1, n])]])
    elif direction == 'd':
        return np.block([[np.zeros([1, n])], [matrix[:-1, :]]])
    elif direction == 'l':
        return np.block([[matrix[:, 1:], np.zeros([n, 1])]])
    elif direction == 'r':
        return np.block([[np.zeros([n, 1]), matrix[:, :-1]]])
    else:
        raise ValueError("The direction must be: 'u', 'd', 'l' or 'r'.")


# =============== #
# === Classes === #
# =============== #

class Distribution(numpy.ndarray):  # A single distribution on the n by n grid

    def __new__(cls, arg):
        if type(arg) == int:
            self = numpy.zeros(shape=(arg, arg)).view(cls)
        elif type(arg) == numpy.ndarray:
            self = (arg / arg.sum()).view(cls)
        else:
            raise ValueError('The input must be int or numpy.ndarray.')

        return self

    def __eq__(self, other, tolerance=1e-10):
        return np.allclose(self, other, rtol=tolerance, atol=tolerance)

    def update(self, tensor):
        s, u, d, l, r = tensor * self
        self[:] = s + shift(u, 'u') + shift(d, 'd') + shift(l, 'l') + shift(r, 'r')

    def distance_from_stationarity(self, tensor):
        return np.abs(tensor.stationary_dist() - self).sum() * .5

    def snap(self, mode='show', file='dummy', dpi=250):
        plt.figure(dpi=dpi)
        plt.axis('off')
        plt.imshow(self)
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(f'{file}.png', bbox_inches='tight')
            plt.close()


class DTensor(numpy.ndarray):

    def __new__(cls, arg):
        if type(arg) == int:
            self = numpy.ndarray(shape=(5, arg, arg)).view(cls)
        elif type(arg) == numpy.ndarray:
            self = arg.view(cls)
        else:
            raise ValueError('The input must be int or numpy.ndarray')

        return self

    def transpose(self):
        s, u, d, l, r = self
        return np.block([[[s]], [[shift(d, 'd')]], [[shift(u, 'u')]], [[shift(r, 'r')]], [[shift(l, 'l')]]])

    def show(self, idcs=range(5), fixed_range=None):
        fig, axs = plt.subplots(1, len(idcs))
        [ax.axis('off') for ax in axs]
        if fixed_range is None:
            [axs[i].imshow(self[idcs[i]]) for i in range(len(idcs))]
        else:
            [axs[i].imshow(self[idcs[i]], vmin=fixed_range[0], vmax=fixed_range[1]) for i in range(len(idcs))]
        plt.show()


class TransitionDTensor(DTensor):  # Parent for all random mixing strategies on the grid

    @classmethod
    def leak(cls, tt):
        s, u, d, l, r = tt
        return u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum()

    @classmethod
    def div_free(cls, tt):
        dim = tt.shape[1:]
        in_flow = tt[0] + shift(tt[1], 'u') + shift(tt[2], 'd') + shift(tt[3], 'l') + shift(tt[4], 'r')

        return in_flow.view(Distribution) == np.ones(dim).view(Distribution)

    def __new__(cls, tt):
        tt /= tt.sum(0)

        if cls.leak(tt) > 0:
            raise ValueError('The transition matrix has leaks')
        if not cls.div_free(tt):
            warnings.warn('Your transition tensor is not divergence free', Warning)

        self = tt.view(cls)
        return self

    @abstractmethod
    def stationary_dist(self):
        pass


class RandomEnvironment(TransitionDTensor):  # iid environment

    def __new__(cls, n):
        tt = np.random.random([4, n, n])

        # removing leaks
        tt[0, 0, :] = 0
        tt[1, -1, :] = 0
        tt[2, :, 0] = 0
        tt[3, :, -1] = 0

        # creating the tensor with 1/2 laziness
        tt = np.block([[[np.ones([n, n])]], [[tt / tt.sum(0)]]]) / 2

        self = tt.view(cls)
        return self

    def stationary_dist(self):
        raise NotImplementedError('This is a hard calculation and would, hence we did not bother with the '
                                  'implementation')


class UpwardDrift(TransitionDTensor):  # iid environment

    def __new__(cls, n, diff=0):
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

        self = tt.view(cls)
        return self

    def stationary_dist(self):
        dim = self.shape[1]

        d = np.block([[np.ones(dim) / dim], [np.zeros([dim - 1, dim])]])
        return Distribution(d)


class DivFreeDTensor(TransitionDTensor):

    def __new__(cls, tt):
        self = np.asarray(tt).view(cls)
        return self

    def stationary_dist(self):
        return Distribution(np.ones(self.shape[1:]))

    def helmholtz(self):
        transposed = self.transpose()

        sym = (self + transposed) / 2
        anti = (self - transposed) / 2
        return sym, anti

    def terrain(self):
        anti = (self - self.transpose()) / 2


class LazyRandomWalk(DivFreeDTensor):

    def __new__(cls, n):
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

        self = (tt / tt.sum(axis=0)).view(cls)
        return self


class Swirl(DivFreeDTensor):

    def __new__(cls, depth, diff=.125, alternating=True, flow_rev=0):
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
        self = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]]).view(cls)
        return self


class BalazsFlow(DivFreeDTensor):

    def __new__(cls, depth, diff=.125, alternating=True):
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
        self = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]]).view(cls)
        return self


class TamasFlow(DivFreeDTensor):

    def __init__(self, tt, alternating=True):
        super().__init__(tt)
