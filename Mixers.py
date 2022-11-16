import numpy as np
import matplotlib.pyplot as plt


# ============= #
# === Utils === #
# ============= #


def shift(matrix, direction):  # Shifting the entries of matrices
    idx = ['u', 'd', 'l', 'r'].index(direction)
    shft = idx % 2
    axis = idx // 2

    return np.roll(matrix, shift=shft * 2 - 1, axis=axis)


class PropertyError(Exception):
    pass


class Terrain(np.ndarray):

    def __eq__(self, other, tolerance=1e-10):
        return np.allclose(self, other, rtol=tolerance, atol=tolerance)

    def __ne__(self, other):
        return not self == other

    def snap(self, mode='show', file='dummy', dpi=250, vmax=None):
        plt.figure(dpi=dpi)
        plt.axis('off')
        plt.imshow(self, vmin=0, vmax=vmax)
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(f'{file}.png', bbox_inches='tight')
            plt.close()


class Distribution(Terrain):  # Distribution on the n by n lattice

    def __new__(cls, arg):
        match type(arg):
            case np.ndarray:
                self = arg.view(cls)
            case _:
                raise ValueError('The input must be numpy.ndarray')

        if self.sum() != 1 or (self < 0).any():
            raise PropertyError('Distribution must be non-negative with sum 1.')

        return self

    def update(self, tensor):
        s, u, d, l, r = tensor * self
        self[:] = s + shift(u, 'u') + shift(d, 'd') + shift(l, 'l') + shift(r, 'r')

    def distance_from_stationarity(self, tensor):
        return np.abs(tensor.stationary_dist() - self).sum() * .5


# ======================== #
# === Property Classes === #
# ======================== #

class GridTensor(np.ndarray):  # Parent class for all matrices on the edges of an n by n lattice

    @staticmethod
    def test(tensor):
        a, b, c = tensor.shape
        if a != 5 or b != c:
            raise PropertyError('GridTensor must have dimensions (5,n,n).')

    def __new__(cls, arg):
        match type(arg):
            case np.ndarray:
                self = arg.view(cls)
            case _:
                raise ValueError('The input must be numpy.ndarray')

        for cl in cls.__bases__:
            try:
                cl.test(self)
            except AttributeError:
                continue

        return self

    def transpose(self):
        s, u, d, l, r = self
        return np.block([[[s]], [[shift(d, 'd')]], [[shift(u, 'u')]], [[shift(r, 'r')]], [[shift(l, 'l')]]])

    def show(self, idcs=range(5), fixed_range=None):
        fig, axs = plt.subplots(1, len(idcs), dpi=250)
        [ax.axis('off') for ax in axs]
        if fixed_range is None:
            [axs[i].imshow(self[idcs[i]]) for i in range(len(idcs))]
        else:
            [axs[i].imshow(self[idcs[i]], vmin=fixed_range[0], vmax=fixed_range[1]) for i in range(len(idcs))]
        plt.show()


class TransitionTensor(GridTensor):  # Parent class for all random mixing strategies on the grid

    @staticmethod
    def test(tt):
        dim = tt.shape[1:]
        sum_condition = tt.sum(0).view(Terrain) != np.ones(dim).view(Terrain)
        pos_condition = (tt < 0).any()

        if sum_condition or pos_condition:
            raise PropertyError('Out going weighs must form a distribution in a TransitionTensor.')

    def __new__(cls, tt):
        tt /= tt.sum(0)
        return super().__new__(cls, tt)


class BoxTensor(GridTensor):

    @staticmethod
    def test(tt):  # On the box some transition probabilities must be zero
        s, u, d, l, r = tt
        if u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum():
            raise PropertyError('BoxTensor must have no leaks.')


class DivFreeTensor(GridTensor):  # parent class for all doubly stochastic Markov chains

    @staticmethod
    def test(tt):  # This checks if we have a doubly stochastic matrix
        dim = tt.shape[1:]
        in_flow = tt[0] + shift(tt[1], 'u') + shift(tt[2], 'd') + shift(tt[3], 'l') + shift(tt[4], 'r')

        if in_flow.view(Terrain) != np.ones(dim).view(Terrain):
            raise PropertyError('Inflows must be distributions in a DivFreeTensor.')

    def stationary_dist(self):
        return np.ones(self.shape[1:]).view(Terrain)

    def helmholtz(self):  # Helmholtz decomposition
        transposed = self.transpose()

        sym = (self + transposed) / 2
        anti = (self - transposed) / 2
        return sym, anti

    def terrain(self):  # A representation of the antisymmetric part of the Helmholtz decomposition
        antisymmetric = (self - self.transpose()) / 2

        ter1 = antisymmetric[1, 1:, 1:].view(Terrain)
        temp = ter1
        for _ in range(ter1.shape[0] - 1):
            temp = shift(temp, 'l')
            ter1 += temp

        ter2 = antisymmetric[3, :-1, 1:].view(Terrain)
        temp = ter2
        for _ in range(ter1.shape[0] - 1):
            temp = shift(temp, 'd')
            ter2 += temp

        return ter1, ter2


# =============================== #
# === Mixing Strategy Classes === #
# =============================== #

class LazyRandomWalk(TransitionTensor, BoxTensor, DivFreeTensor):

    def __new__(cls, n):
        tt = np.block([[[4 * np.ones([n, n])]], [[np.ones([4, n, n])]]])

        # removing leaks
        tt[1, 0] -= 1
        tt[2, -1] -= 1
        tt[3, :, 0] -= 1
        tt[4, :, -1] -= 1

        # those leaks are transferred to laziness
        tt[0, 0] += 1
        tt[0, -1] += 1
        tt[0, :, 0] += 1
        tt[0, :, -1] += 1

        tt = (tt / tt.sum(0))
        return tt.view(cls)

    @staticmethod
    def sample_map():
        return [[0, 0]]


class NaiveBalazsFlow(TransitionTensor, BoxTensor, DivFreeTensor):  # A recursively defined strategy

    def __new__(cls, depth, diff=None, alternating=True):
        if diff is None:
            diff = 1 / (2 ** (depth + 1) - 2)

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
        return super().__new__(cls, tt)

    def sample_map(self):
        return [[i, i] for i in range(self.shape[1] // 4, self.shape[1] // 2 + 1)]


class BalazsFlow(TransitionTensor, BoxTensor, DivFreeTensor):  # Another version of the previous class

    def __new__(cls, depth):

        # === upstream === #
        def update_seeds(sp, d1, d2):
            temp = []
            for x, y in sp:
                temp += [(x + d1 * (1 + n * (i // 2)), y + d2 * (1 + n * (i % 2))) for i in range(4)]
            return temp

        def update_up(m, sp, c, f):
            if f == 'jet':
                for x, y in sp:
                    m[x:x + n - 1, y] = c

            elif f == 'in':
                for x, y in sp:
                    m[x, y:y + n - 2] = c

            elif f == 'out':
                for x, y in sp:
                    m[x, y:y + n] = c

        # init
        n = 2 ** (depth + 1) - 2
        diff = .25 / (n - 2)
        up = np.zeros([n, n])
        up[1:, -1] = .5
        up[-1, 1:-1] = diff

        # starting points for updating the flows
        seeds_jet = [(1, -1)]
        seeds_in = [(-1, 1)]
        seeds_out = [(0, 0)]

        for d in np.arange(1, depth)[::-1]:
            n = 2 ** (d + 1) - 2

            # seeds of the lower level
            seeds_jet = update_seeds(seeds_jet, 1, -1)
            seeds_in = update_seeds(seeds_in, -1, 1)
            seeds_out = update_seeds(seeds_out, 1, 1)

            # filling in the probabilities of the level
            update_up(up, seeds_jet, .5, 'jet')
            update_up(up, seeds_out, diff, 'out')
            diff = .25 / (n - 1) - diff
            update_up(up, seeds_in, diff, 'in')

        # === left-, right- and downstream === #
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)

        # === laziness === #
        n = 2 ** (depth + 1) - 2
        stay = np.ones((n, n)) - up - down - left - right

        # === ensemble === #
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        return super().__new__(cls, tt)

    def sample_map(self):
        return [[i, i] for i in range(self.shape[1] // 4, self.shape[1] // 2 + 1)]


class FatBalazsFlow(TransitionTensor, BoxTensor, DivFreeTensor):  # The third version

    def __new__(cls, depth, alpha=.6):
        # === antisymmetric upstream === #
        def stream(n, m):
            retv = np.zeros((n, m))
            for i in range(m):
                retv[range(i + 1, n - i), [i] * (n - 1 - 2 * i)] = 1

            return retv

        def recursion(d):
            if d == 0:
                return np.empty([0, 0])

            prev = recursion(d - 1)
            n = prev.shape[0]
            m = int(n ** alpha) + 1

            curr = np.block([[np.zeros((m, 2 * n))], [prev, prev], [prev, prev], [np.zeros((m, 2 * n))]])
            band = stream(2 * (n + m), m) * .125
            curr = np.block([-band, curr, band[:, ::-1]])

            return curr

        up = recursion(depth) + .125
        up[0, :] = 0

        # === left-, right-, downstream and laziness === #
        left = np.rot90(up)
        down = np.rot90(left)
        right = np.rot90(down)
        lazy = 1 - up - down - left - right

        # === ensemble === #
        tt = np.block([[[lazy]], [[up]], [[down]], [[left]], [[right]]])
        return super().__new__(cls, tt)

    def sample_map(self):
        return [[i, i] for i in range(self.shape[1] // 4, self.shape[1] // 2 + 1)]


class Highway(TransitionTensor, BoxTensor, DivFreeTensor):  # Best mixer so far

    def __new__(cls, k):
        n = 8 * k ** 3  # side length of the grid
        w = 2 * k ** 2  # width of the path

        def directions(entry):
            r = entry // n
            c = entry % n

            # I am truly sorry about these
            cond1 = (r - int(r >= w) < c % (4 * w)) and (r + c % (4 * w) < 4 * w - int(r >= w))
            cond2 = (n - 2 * w > c) and (c >= 2 * w)
            cond3 = (n / 2 - r - int(n / 2 - r <= w) <= (c + 2 * w) % (4 * w)) and (
                    n / 2 - r + (c + 2 * w) % (4 * w) - int(n / 2 - r > w) < 4 * w)

            if cond1:
                return [-1, 1][r < w]

            elif cond2 and cond3:
                return [-1, 1][n / 2 - r > w]

            else:
                return [-2, 2][int(c % (2 * w) >= w)]

        v_directions = np.vectorize(directions)

        dirs = v_directions(np.arange(n * n // 2).reshape([n // 2, n]))
        dirs = np.block([[dirs], [-dirs[::-1, ::-1]]])

        up = ((dirs == 2).astype(int) - shift((dirs == -2).astype(int), 'd')) + 1
        up[0] = 0
        down = up[::-1, ::-1]

        left = ((dirs == 1).astype(int) - shift((dirs == -1).astype(int), 'r')) + 1
        left[:, 0] = 0
        right = left[::-1, ::-1]

        stay = np.ones(up.shape) * 4
        stay[[0, -1]] += 1
        stay[:, [0, -1]] += 1

        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        tt /= tt.sum(0)
        return super().__new__(cls, tt)

    def sample_map(self):
        return [[i, i] for i in range(self.shape[1] // 2 + 1)]


class FractalSwirl(TransitionTensor, DivFreeTensor):

    def __new__(cls, k, q):
        if q <= 0 or q >= 1:
            raise ValueError("Parameter 'q' should be between 0 and 1.")

        n = 2 ** k  # side length of the grid

        # creating the lazy random walk with periodic boundary
        lrw = np.block([[[4 * np.ones([n, n])]], [[np.ones([4, n, n])]]]) / 8

        # creating the drifts
        drifts = np.zeros(lrw.shape)
        for i in range(k):
            idcs = np.nonzero(np.arange(n) % 2 ** (i + 1) // 2 ** i)[0]

            drift_temp = np.zeros([n, n])
            drift_temp[idcs, :] = .25
            drift_temp -= .125
            drift_temp *= q ** i

            drifts[4] += drift_temp
        drifts[1] += np.rot90(drifts[4], k=1)
        drifts[2] += np.rot90(drifts[4], k=3)
        drifts[3] += np.rot90(drifts[4], k=2)

        self = lrw + drifts * (1 - q)
        return super().__new__(cls, self)

    def sample_map(self):
        n = self.shape[1]
        return [[i, i] for i in range(n // 4, 3 * n // 8)]


# ================= #
# === Old Codes === #
# ================= #

class DTensor(np.ndarray):  # Parent class for all matrices on the edges of an n by n lattice

    def __new__(cls, arg):
        if type(arg) == int:
            self = np.ndarray(shape=(5, arg, arg)).view(cls)
        elif type(arg) == np.ndarray:
            self = arg.view(cls)
        else:
            raise ValueError('The input must be int or numpy.ndarray')

        return self

    def transpose(self):
        s, u, d, l, r = self
        return np.block([[[s]], [[shift(d, 'd')]], [[shift(u, 'u')]], [[shift(r, 'r')]], [[shift(l, 'l')]]])

    def show(self, idcs=range(5), fixed_range=None):
        fig, axs = plt.subplots(1, len(idcs), dpi=250)
        [ax.axis('off') for ax in axs]
        if fixed_range is None:
            [axs[i].imshow(self[idcs[i]]) for i in range(len(idcs))]
        else:
            [axs[i].imshow(self[idcs[i]], vmin=fixed_range[0], vmax=fixed_range[1]) for i in range(len(idcs))]
        plt.show()


class TransitionDTensorTorus(DTensor):  # Parent class for all random mixing strategies on the grid

    def __new__(cls, tt):
        tt /= tt.sum(0)
        return tt.view(cls)


class TransitionDTensorBox(TransitionDTensorTorus):

    @staticmethod
    def leak(tt):  # On the box some transition probabilities must be zero
        s, u, d, l, r = tt
        return u[:1, :].sum() + d[-1:, :].sum() + l[:, :1].sum() + r[:, -1:].sum()

    def __new__(cls, tt):
        if cls.leak(tt) > 0:
            raise ValueError('Your transition tensor leaks. If it is supposed to be on the torus inherit from '
                             'TransitionDTensorTorus instead of TransitionDTensorBox')
        return tt.view(cls)


class DivFreeDTensor(TransitionDTensorTorus):  # parent class for all doubly stochastic Markov chains

    @staticmethod
    def div_free(tt):  # This checks if we have a doubly stochastic matrix
        dim = tt.shape[1:]
        in_flow = tt[0] + shift(tt[1], 'u') + shift(tt[2], 'd') + shift(tt[3], 'l') + shift(tt[4], 'r')

        return in_flow.view(Terrain) == np.ones(dim).view(Terrain)

    def __new__(cls, tt):
        if not cls.div_free(tt):
            raise ValueError('Your transition tensor is not divergence free.')

        return tt.view(cls)

    def stationary_dist(self):
        return Terrain(np.ones(self.shape[1:]))

    def helmholtz(self):  # Helmholtz decomposition
        transposed = self.transpose()

        sym = (self + transposed) / 2
        anti = (self - transposed) / 2
        return sym, anti

    def terrain(self):  # A representation of the antisymmetric part of the Helmholtz decomposition
        anti = (self - self.transpose()) / 2

        ter1 = anti[1, 1:, 1:].view(Terrain)
        temp = ter1
        for _ in range(ter1.shape[0] - 1):
            temp = shift(temp, 'l')
            ter1 += temp

        ter2 = anti[3, :-1, 1:].view(Terrain)
        temp = ter2
        for _ in range(ter1.shape[0] - 1):
            temp = shift(temp, 'd')
            ter2 += temp

        if ter1 != ter2:
            raise ValueError('Something is wrong check the method DivFreeDTensor.terrain()')

        return ter1


class RandomEnvironment(TransitionDTensorBox):  # iid uniform transitions on each edge

    def __new__(cls, n):
        tt = np.random.random([4, n, n])

        # removing leaks
        tt[0, 0, :] = 0
        tt[1, -1, :] = 0
        tt[2, :, 0] = 0
        tt[3, :, -1] = 0

        # creating the tensor with 1/2 laziness
        tt = np.block([[[np.ones([n, n])]], [[tt / tt.sum(0)]]]) / 2

        return super().__new__(cls, tt)

    def stationary_dist(self):
        raise NotImplementedError('This is not our scope, hence we did not bother with the implementation')

    def sample_map(self):
        raise NotImplementedError('This is not our scope, hence we did not bother with the implementation')


class UpwardDrift(TransitionDTensorBox):

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

        return super().__new__(cls, tt)

    def stationary_dist(self):
        dim = self.shape[1]

        d = np.block([[np.ones(dim) / dim], [np.zeros([dim - 1, dim])]])
        return Terrain(d)

    def sample_map(self):
        return [[0, self.shape[0] - 1]]


class Swirl(TransitionDTensorBox, DivFreeDTensor):  # A whirlpool around the center of the grid

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
        tt = np.block([[[stay]], [[up]], [[down]], [[left]], [[right]]])
        return super().__new__(cls, tt)

    @staticmethod
    def sample_map():
        return [[0, 0]]


class AntiSymmetric(DTensor):  # Parent class for all antisymmetric matrices

    def __new__(cls, terrain):  # we only used this class of matrices for the Helmholtz decomposition
        n, m = terrain.shape

        u = np.block([np.zeros((n, 1)), terrain]) - np.block([terrain, np.zeros((n, 1))])
        d = np.block([[-u], [np.zeros((1, n + 1))]])
        u = np.block([[np.zeros((1, n + 1))], [u]])

        l = np.block([[terrain], [np.zeros((1, m))]]) - np.block([[np.zeros((1, m))], [terrain]])
        r = np.block([[-l, np.zeros((m + 1, 1))]])
        l = np.block([[np.zeros((m + 1, 1)), l]])

        self = np.block([[[u]], [[d]], [[l]], [[r]]]).view(cls)
        return self
