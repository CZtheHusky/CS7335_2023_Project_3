"""
Kernel Mean Matching
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
"""


import sklearn.metrics
from cvxopt import matrix, solvers
from utils import *
from joblib import Parallel, delayed


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta




def run(src_domain, tar_domain):
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    print("-------------------------------------------")
    print('Source:', src_domain, Xs.shape, Ys.shape)
    print('Target:', tar_domain, Xt.shape, Yt.shape)
    kmm = KMM(kernel_type='rbf', B=10)
    beta = kmm.fit(Xs, Xt)
    Xs_new = beta * Xs
    results = Parallel(n_jobs=4)(
        delayed(svm_classify)(source_data, Ys, Xt, Yt, norm=norm) for norm in
        [True, False] for source_data in [Xs_new, Xs])
    print("---------------------------")
    print("Norm On")
    print("SVM with KMM features:", results[0])
    print("SVM with original features:", results[1])
    print("Performance gain with KMM:", results[0] - results[1])
    print("---------------------------")
    print("Norm Off")
    print("SVM with KMM features:", results[2])
    print("SVM with original features:", results[3])
    print("Performance gain with KMM:", results[2] - results[3])
    print("---------------------------")
    print("-------------------------------------------")


if __name__ == "__main__":
    run('Art', 'RealWorld')
    run("Clipart", "RealWorld")
