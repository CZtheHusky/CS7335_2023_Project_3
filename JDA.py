# encoding=utf-8
"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from utils import *
from joblib import Parallel, delayed


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        Xs_news = []
        Xt_news = []
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            Xs_news.append(Xs_new)
            Xt_news.append(Xt_new)
        return Xs_news, Xt_news
    
    
def JDA_core(Xs,Ys, Xt, jda_args, id):
    try:
        jda = JDA(kernel_type=jda_args["kernel_type"], dim=jda_args['dim'], lamb=jda_args['lamb'], gamma=jda_args['gamma'])
        Xs_news, Xt_news = jda.fit(Xs, Ys, Xt)
    except Exception as e:
        return False, Xs, Xt, id
    return True, Xs_news, Xt_news, id



def JDA_search(src_domain, tar_domain, jda_arg_list):
    jda_process = min(128, len(jda_arg_list))
    Xs, Ys, Xt, Yt = load_data(src_domain, tar_domain)
    JDA_results = Parallel(n_jobs=jda_process)(delayed(JDA_core)(Xs, Ys, Xt, jda_args, id) for id, jda_args in enumerate(jda_arg_list))
    JDA_cleaned = [result for result in JDA_results if result[0]]
    svm_process = min(128, len(JDA_cleaned) * 10)
    results = Parallel(n_jobs=svm_process)(delayed(svm_jda)(Xs_new, Ys, Xt_new, Yt, norm=True, id=cleaned[3]) for cleaned in JDA_cleaned for Xs_new, Xt_new in zip(cleaned[1], cleaned[2]))
    results = np.array(results, dtype=np.float32)
    results = results[results[:, 1].argsort()]
    acc_res = results[:, 0].reshape(-1, 10)
    best_accs = np.max(acc_res, axis=1)
    best_arg = jda_arg_list[JDA_cleaned[np.argmax(best_accs)][3]]
    best_acc = max(best_accs)
    to_print = ""
    to_print += "-------------------------------------------\n"
    to_print += f"Source: {src_domain} Target: {tar_domain}\n"
    to_print += f"Best Args: {best_arg}\n"
    to_print += f"Best Performance: {best_acc}\n"
    to_print += "-------------------------------------------\n"
    print(to_print)
    return best_acc, best_arg, to_print


if __name__ == "__main__":
    domain_pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld')]
    kernel_types = ['primal', 'linear', 'rbf']
    dims = [10, 30, 50]
    lambs = [0.1, 1, 10]
    gammas = [0.1, 1, 10]
    jda_arg_list = []
    for kernel_type in kernel_types:
        for dim in dims:
            for lamb in lambs:
                for gamma in gammas:
                    jda_arg_list.append({"kernel_type": kernel_type, "dim": dim, "lamb": lamb, "gamma": gamma})
    JDA_results =  [JDA_search('Art', 'RealWorld', jda_arg_list), JDA_search("Clipart", "RealWorld", jda_arg_list)]
    for acc, arg, log in JDA_results:
        print(log)