from numpy import log, abs, zeros, inf, exp, array, random, ones, sign, mean, spacing, add
from thunder.utils import distances
from thunder.utils.distances import Distance

class TSNE(object):

    def __init__(self, nodims=2, maxiter=10, metric="euclidean", yinit=None):
        self.momentum = 0.5                 # initial momentum
        self.final_momentum = 0.8           # value to which momentum is changed
        self.mom_switch_iter = 250          # iteration momentum changed
        self.stop_lying_iter = 125          # iteration we stop lying about p-values
        self.maxiter = maxiter               # maximum number of iterations
        self.epsilon = 500                  # initial learning rate
        self.mingain = .01                 # minimum gain for delta_bar_delta
        self.lie_multiplier = 4             # lie multiplier
        self.old_cost = 1e10                # initial cost
        self.rel_tol = 1e-4                 # relative tolerance
        self.u = 15  # perplexity
        self.tol = 1e-4
        self.logu = log(self.u)
        self.beta = 1
        self.nodims = nodims
        self.maxiter = maxiter
        self.metric = metric
        self.yinit = yinit


    @staticmethod
    def distmat(data, metric="euclidean"):
        
        dist = Distance(metric)
        return data.cartesian(data).map(lambda ((a, b), (c, d)): (a + c, dist.get(b, d)))


    def hbetafind(self, c, beta=float(1)):
        betamin = -inf
        betamax = inf
        c = sorted(c)
        d = [x[1] for x in c]
        d0 = [x for x in [y if y != 0 else inf for y in d]]
        p = exp(-array(d0).copy() * beta)
        psum = sum(p)
        h = log(psum) + beta * sum(array(d) * p) / psum
        p /= psum
        hdiff = h - self.logu
        tries = 0
        while abs(hdiff) > self.tol and tries < 50:
            # if not, increase or decrease precision
            if hdiff > 0:
                betamin = float(beta)
                if betamax == inf or betamax == -inf:
                    beta = float(beta) * 2
                else:
                    beta = float(beta + betamax) / 2
            else:
                betamax = float(beta)
                if betamin == inf or betamin == -inf:
                    beta = float(beta) / 2
                else:
                    beta = float(beta + betamin) / 2
            p = exp(-array(d0).copy() * float(beta))
            psum = sum(p)
            h = log(psum) + float(beta * sum(array(d) * p) / psum)
            p /= psum
            hdiff = h - self.logu
            tries += 1

        return p

    def train(self, data):
        n = data.count()
        dist = TSNE.distmat(data, self.metric)
        distbykey = dist.map(lambda ((k1, k2), v): (k1, (k2, v))).groupByKey()
        prows = distbykey.map(lambda ((k1, b)): (k1, self.hbetafind(b)))
        pnorm = prows.flatMap(lambda (k, v): map(lambda (a, b): ((k, a), b), zip(range(1, n+1), v.tolist())))
        pnorm.cache()

        if self.yinit is None:
            random.seed(42)
            y = 0.0001 * random.random((n, self.nodims))
        else:
            y = yinit

        y_incs = zeros((n, self.nodims))
        gains = ones((n, self.nodims))
        cost = zeros(self.maxiter)

        # define inline functions           
        squareddist = lambda d, q: sum((d - q) ** 2)
        invsquare = lambda k1, k2, v : 0 if k1 == k2 else 1/(v + 1)**2
        logratio = lambda p, q1, q2: p * log((p + spacing(1)) / (squareddist(q1, q2) + spacing(1)))

        for iter in range(self.maxiter):

            # create an rdd from, and also broadcast, the current data points
            y_rdd = data.context.parallelize(map(lambda (a, b): ((a,), b), zip(range(1, n+1), y)), 10)
            y_bc = data.context.broadcast(y)

            # compute the normalizing constant through a cartesian product and sum    
            y_d = TSNE.distmat(y_rdd)
            z = y_d.map(lambda (k, v): invsquare(k[0], k[1], v)).sum()

            # use the broadcast version of y to compute l without doing a join
            l = pnorm.map(lambda (k, x): (k, (x, invsquare(k[0], k[1], squareddist(y_bc.value[k[0] - 1, :], y_bc.value[k[1] - 1, :])))))\
                .mapValues(lambda (x, y): ((x - y/z) * y))

            # compute gradient through a single reduceByKey on l
            # might need to flip k1 and k2 in key assignment before reduceByKey (via comparison to the matlab version), double check!
            y_grads = l.map(lambda ((k1, k2), v): (k1, v * (-y_bc.value[k2-1, :] + y_bc.value[k1-1, :])))\
                .reduceByKey(add).collect()
            y_grads = array(map(lambda (k, v): v, sorted(y_grads)))
            
            # update the gradients based on the gain and momentum
            gains = (gains+.2) * (sign(y_grads) != sign(y_incs)) + (gains*.8) * (sign(y_grads) == sign(y_incs))
            gains[gains > self.mingain] = self.mingain
            y_incs = self.momentum * y_incs - self.epsilon * (gains * y_grads)
            y += y_incs
            y -= mean(y, 0)

            # estimate the cost
            cost[iter] = pnorm.map(lambda ((k1, k2), v): logratio(v, y_bc.value[k1-1, :], y_bc.value[k2-1, :])).reduce(add)

            if iter == self.mom_switch_iter:
                self.momentum = self.final_momentum

        self.y = y
        self.cost = cost

        return self

