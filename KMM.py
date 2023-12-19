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
    
    
    
def KMM_core(Xs, Xt, b):
    try:
        kmm = KMM(kernel_type='rbf', B=b)
        beta = kmm.fit(Xs, Xt)
        Xs_new = beta * Xs
    except Exception as e:
        return False, Xs, b
    return True, Xs_new, b



def KMM_search(src_domain, tar_domain, betas):
    kmm_process = min(64, len(betas))
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    KMM_results = Parallel(n_jobs=kmm_process)(delayed(KMM_core)(Xs, Xt, beta) for beta in betas)
    KMM_cleaned = [result for result in KMM_results if result[0]]
    svm_process = min(64, len(KMM_cleaned))
    results = Parallel(n_jobs=svm_process)(delayed(svm_classify)(cleaned[1], Ys, Xt, Yt, norm=True) for cleaned in KMM_cleaned)
    best_beta = KMM_cleaned[np.argmax(results)][2]
    best_acc = max(results)
    to_print = ""
    to_print += "-------------------------------------------\n"
    to_print += f"Source: {src_domain} Target: {tar_domain}\n"
    to_print += f"Best Beta: {best_beta}\n"
    to_print += f"Best Performance: {best_acc}\n"
    to_print += "-------------------------------------------\n"
    print(to_print)
    return best_acc, best_beta, to_print


def run(src_domain, tar_domain, beta=1):
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    Xs_new = KMM_core(Xs, Xt, beta)
    results = Parallel(n_jobs=4)(
        delayed(svm_classify)(source_data, Ys, Xt, Yt, norm=norm) for norm in
        [True, False] for source_data in [Xs_new, Xs])
    to_print = ""
    to_print += "-------------------------------------------\n"
    to_print += f"Source: {src_domain} Target: {tar_domain}\n"
    to_print += f"Beta: {beta}\n"
    to_print += "Norm On\n"
    to_print += f"SVM with KMM features: {results[0]}\n"
    to_print += f"SVM with original features: {results[1]}\n"
    to_print += f"Performance gain with KMM: {results[0] - results[1]}\n"
    to_print += "Norm Off\n"
    to_print += f"SVM with KMM features: {results[2]}\n"
    to_print += f"SVM with original features: {results[3]}\n"
    to_print += f"Performance gain with KMM: {results[2] - results[3]}\n"
    to_print += "-------------------------------------------\n"
    print(to_print)
    return results


if __name__ == "__main__":
    domain_pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld')]
    # baseline_results = Parallel(n_jobs=2)(delayed(baseline)(src, tar) for src, tar in domain_pairs)
    betas = np.linspace(0.1, 128, 128)
    KMM_results =  [KMM_search('Art', 'RealWorld', betas), KMM_search("Clipart", "RealWorld", betas)]
    for (src, tar), (acc, beta, log) in zip(domain_pairs, KMM_results):
        print("Source:", src, "Target:", tar, "Best Beta:", beta, "Best Performance:", acc)