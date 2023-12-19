# encoding=utf-8
"""
    Created on 17:25 2018/11/13
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from utils import *
import pygensvd

class GFK:
    def __init__(self, dim=200):
        '''
        Init func
        :param dim: dimension after GFK
        '''
        self.dim = dim
        self.eps = 1e-20

    def fit(self, Xs, Xt, norm_inputs=None):
        '''
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        '''
        if norm_inputs:
            source, mu_source, std_source = self.znorm(Xs)
            target, mu_target, std_target = self.znorm(Xt)
        else:
            mu_source = np.zeros(shape=(Xs.shape[1]))
            std_source = np.ones(shape=(Xs.shape[1]))
            mu_target = np.zeros(shape=(Xt.shape[1]))
            std_target = np.ones(shape=(Xt.shape[1]))
            source = Xs
            target = Xt

        Ps = self.train_pca(source, mu_source, std_source, 0.99)
        Pt = self.train_pca(target, mu_target, std_target, 0.99)
        Ps = np.hstack((Ps, scipy.linalg.null_space(Ps.T)))
        Pt = Pt[:, :self.dim]
        N = Ps.shape[1]
        dim = Pt.shape[1]
        # Principal angles between subspaces
        QPt = np.dot(Ps.T, Pt)

        A = QPt[0:dim, :].copy()
        B = QPt[dim:, :].copy()

        # Equation (2)
        Gam, Sig, V, V1, V2 = pygensvd.gsvd(A, B, full_matrices=True, extras='uv')
        V2 = -V2

        theta = np.arccos(Gam)
        # Equation (6)
        B1 = np.diag(0.5 * (1 + (np.sin(2 * theta) / (2. * np.maximum(theta, 1e-20)))))
        B2 = np.diag(0.5 * ((np.cos(2 * theta) - 1) / (2 * np.maximum(theta, self.eps))))
        B3 = B2
        B4 = np.diag(0.5 * (1 - (np.sin(2 * theta) / (2. * np.maximum(theta, self.eps)))))

        # Equation (9) of the suplementary matetial
        delta1_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta1_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta1 = np.vstack((delta1_1, delta1_2))

        delta2_1 = np.hstack((B1, B2, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_2 = np.hstack((B3, B4, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_3 = np.zeros(shape=(N - 2 * dim, N))
        delta2 = np.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta3_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta3 = np.vstack((delta3_1, delta3_2)).T

        delta = np.linalg.multi_dot([delta1, delta2, delta3])
        G = np.linalg.multi_dot([Ps, delta, Ps.T])
        sqG = np.real(scipy.linalg.fractional_matrix_power(G, 0.5))
        Xs_new, Xt_new = np.dot(sqG, Xs.T).T, np.dot(sqG, Xt.T).T
        return G, Xs_new, Xt_new

    def principal_angles(self, Ps, Pt):
        """
        Compute the principal angles between source (:math:`P_s`) and target (:math:`P_t`) subspaces in a Grassman which is defined as the following:

        :math:`d^{2}(P_s, P_t) = \sum_{i}( \theta_i^{2} )`,

        """
        # S = cos(theta_1, theta_2, ..., theta_n)
        _, S, _ = np.linalg.svd(np.dot(Ps.T, Pt))
        thetas_squared = np.arccos(S) ** 2

        return np.sum(thetas_squared)

    def train_pca(self, data, mu_data, std_data, subspace_dim):
        '''
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param mu_data: mu
        :param std_data: std
        :param subspace_dim: dim
        :return: a wrapped machine object
        '''
        if isinstance(subspace_dim, float):
            pca = PCA(n_components=subspace_dim, svd_solver='full')
        else:
            pca = PCA(n_components=subspace_dim)
        pca.fit((data - mu_data) / std_data)
        return pca.components_.T

    def znorm(self, data):
        """
        Z-Normaliza
        """
        mu = np.average(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mu) / std
        return data, mu, std

    def subspace_disagreement_measure(self, Ps, Pt, Pst):
        """
        Get the best value for the number of subspaces
        For more details, read section 3.4 of the paper.
        **Parameters**
          Ps: Source subspace
          Pt: Target subspace
          Pst: Source + Target subspace
        """

        def compute_angles(A, B):
            _, S, _ = np.linalg.svd(np.dot(A.T, B))
            S[np.where(np.isclose(S, 1, atol=self.eps) == True)[0]] = 1
            return np.arccos(S)

        max_d = min(Ps.shape[1], Pt.shape[1], Pst.shape[1])
        alpha_d = compute_angles(Ps, Pst)
        beta_d = compute_angles(Pt, Pst)
        d = 0.5 * (np.sin(alpha_d) + np.sin(beta_d))
        return np.argmax(d)


def GFK_core(Xs, Xt, dim=200):
    try:
        gfk = GFK(dim=dim)
        G, Xs_new, Xt_new = gfk.fit(Xs, Xt, True)
    except Exception as e:
        return False, Xs, Xt, dim
    return True, Xs_new, Xt_new, dim
    
def GFK_search(src_domain, tar_domain, dims):
    gfk_processes = min(64, len(dims))
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    GFK_results = Parallel(n_jobs=gfk_processes)(delayed(GFK_core)(Xs, Xt, dim) for dim in dims)
    GFK_cleaned = [result for result in GFK_results if result[0]]
    svm_process = min(64, len(GFK_cleaned))
    results = Parallel(n_jobs=svm_process)(delayed(svm_classify)(cleaned[1], Ys, cleaned[2], Yt, norm=True) for cleaned in GFK_cleaned)
    best_dim = GFK_cleaned[np.argmax(results)][3]
    best_acc = max(results)
    to_print = ""
    to_print += "-------------------------------------------\n"
    to_print += f"Source: {src_domain} Target: {tar_domain}\n"
    to_print += f"Best Dim: {best_dim}\n"
    to_print += f"Best Performance: {best_acc}\n"
    to_print += "-------------------------------------------\n"
    print(to_print)
    return best_acc, best_dim


def run(src_domain, tar_domain, ):
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    print("-------------------------------------------")
    print('Source:', src_domain, 'Target:', tar_domain)
    gfk = GFK(dim=200)
    G, Xs_new, Xt_new = gfk.fit(Xs, Xt, True)
    results = Parallel(n_jobs=2)([
        delayed(svm_classify)(Xs_new, Ys, Xt_new, Yt, norm=False),
        delayed(svm_classify)(Xs, Ys, Xt, Yt, norm=False)
    ])
    print("SVM with GFK features:", results[0])
    print("SVM with original features:", results[1])
    print("Performance gain with GFK:", results[0] - results[1])
    print("-------------------------------------------")

if __name__ == "__main__":
    dims = [10, 25, 50, 100, 200, 400, 800]
    domain_pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld')]
    GFK_results =  [GFK_search('Art', 'RealWorld', dims), GFK_search("Clipart", "RealWorld", dims)]
    for (src, tar), (acc, dim) in zip(domain_pairs, GFK_results):
        print(f"Source: {src} Target: {tar} Best Performance: {acc} Best Dim: {dim}")
    