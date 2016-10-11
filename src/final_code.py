
# coding: utf-8

# In[1]:

import glob # use your path
import pandas as pd
from sklearn.cluster import KMeans
import scipy
import numpy as np


# In[ ]:

# # Copyright Mathieu Blondel December 2011
# # License: BSD 3 clause

# import numpy as np
# import pylab as pl

# from sklearn.base import BaseEstimator
# from sklearn.utils import check_random_state
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster import KMeans as KMeansGood
# from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
# from sklearn.datasets.samples_generator import make_blobs

# ##############################################################################
# # Generate sample data
# np.random.seed(0)

# batch_size = 45
# centers = [[1, 1], [-1, -1], [1, -1]]
# n_clusters = len(centers)
# X, labels_true = make_blobs(n_samples=1200, centers=centers, cluster_std=0.3)

# class KMeans(BaseEstimator):

#     def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
#         self.k = k
#         self.max_iter = max_iter
#         self.random_state = random_state
#         self.tol = tol

#     def _e_step(self, X):
#         self.labels_ = euclidean_distances(X, self.cluster_centers_,
#                                      squared=True).argmin(axis=1)

#     def _average(self, X):
#         return X.mean(axis=0)

#     def _m_step(self, X):
#         X_center = None
#         for center_id in range(self.k):
#             center_mask = self.labels_ == center_id
#             if not np.any(center_mask):
#                 # The centroid of empty clusters is set to the center of
#                 # everything
#                 if X_center is None:
#                     X_center = self._average(X)
#                 self.cluster_centers_[center_id] = X_center
#             else:
#                 self.cluster_centers_[center_id] = \
#                     self._average(X[center_mask])

#     def fit(self, X, y=None):
#         n_samples = X.shape[0]
#         vdata = np.mean(np.var(X, 0))

#         random_state = check_random_state(self.random_state)
#         self.labels_ = random_state.permutation(n_samples)[:self.k]
#         self.cluster_centers_ = X[self.labels_]

#         for i in xrange(self.max_iter):
#             centers_old = self.cluster_centers_.copy()

#             self._e_step(X)
#             self._m_step(X)

#             if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
#                 break

#         return self

# class KMedians(KMeans):

#     def _e_step(self, X):
#         self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

#     def _average(self, X):
#         return np.median(X, axis=0)

# class FuzzyKMeans(KMeans):

#     def __init__(self, k, m=2, max_iter=100, random_state=0, tol=1e-4):
#         """
#         m > 1: fuzzy-ness parameter
#         The closer to m is to 1, the closter to hard kmeans.
#         The bigger m, the fuzzier (converge to the global cluster).
#         """
#         self.k = k
#         assert m > 1
#         self.m = m
#         self.max_iter = max_iter
#         self.random_state = random_state
#         self.tol = tol

#     def _e_step(self, X):
#         D = 1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)
#         D **= 1.0 / (self.m - 1)
#         D /= np.sum(D, axis=1)[:, np.newaxis]
#         # shape: n_samples x k
#         self.fuzzy_labels_ = D
#         self.labels_ = self.fuzzy_labels_.argmax(axis=1)

#     def _m_step(self, X):
#         weights = self.fuzzy_labels_ ** self.m
#         # shape: n_clusters x n_features
#         self.cluster_centers_ = np.dot(X.T, weights).T
#         self.cluster_centers_ /= weights.sum(axis=0)[:, np.newaxis]

#     def fit(self, X, y=None):
#         n_samples, n_features = X.shape
#         vdata = np.mean(np.var(X, 0))

#         random_state = check_random_state(self.random_state)
#         self.fuzzy_labels_ = random_state.rand(n_samples, self.k)
#         self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
#         self._m_step(X)

#         for i in xrange(self.max_iter):
#             centers_old = self.cluster_centers_.copy()

#             self._e_step(X)
#             self._m_step(X)

#             if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
#                 break

#         return self


# In[2]:

def preparedata(file1,zh):
    data2 = pd.DataFrame()
    for i in file1:
        print i
        data = pd.read_csv(i)
        data.drop(['Date','Houseid'], axis=1, inplace=True)
        data = data.fillna(0)
        data1 = pd.DataFrame(data.iloc[zh])
        data2 = pd.concat([data1,data2],axis=1)
    print data2, "data2"
    return data2
        
def kmeansdata(data2):
    da2 = data2.transpose()
    k_mea= KMeans(init='k-means++', n_clusters=47)
#     k_mea= KMeans(k=47)
    barc = k_mea.fit(da2)
    fdf2 = pd.DataFrame(barc.cluster_centers_)
    fdf3 = fdf2.transpose()
    print fdf3, "fdf3 inside function"
    return fdf3

def takeabs(fdf3,actual):
    delta_t = 60
    jh = 1440/delta_t
    ans = []
    for restf in fdf3.columns:
        a = fdf3[restf]
        for shift in range(jh):
            ans.append(scipy.absolute(scipy.roll(a,shift) - actual).sum())
    indexloc = ans.index(min(ans))
    pstar = indexloc/jh
    jstar= indexloc%jh
    adf=fdf3[pstar]
    adff=np.roll(adf,jstar)
    adff1=pd.Series(adff)
    print adff1,"pd.Series(adff)"
    return adff1

def mergdat(mergpat,bestpat):
    bestpat = pd.concat([mergpat,bestpat], axis=1)
    return bestpat

ite = 0
words = ['air','dryer','furnace','lights_plugs','refridgerator']
# words = ['air','dryer']
for readit in range(0,14400,1440):
# for readit in range(0,2880,1440):
    testf = pd.read_csv("26__weekday_out.csv", skiprows=readit , nrows=1440)
    bestpatmp = pd.DataFrame()
    actual = testf.iloc[:,7]
    print actual
    for  i in words:
        ghj = '*_'+i+'_weekday_out.csv'
        allFiles = glob.glob(ghj)
        data3 = preparedata(allFiles,ite)
        kmpat = kmeansdata(data3)
        bestpat = takeabs(kmpat,actual)
        actual = actual.subtract(bestpat)
        print actual, "actual"
        bestpatmp = mergdat(bestpatmp,bestpat)
    bestpatmp.to_csv('solution.csv', mode='a', index = False, index_label = False, header=False)
    ite += 1


# In[3]:

r1 = pd.read_csv("solution.csv", header=-1)
r1.columns = ["a","b"]
chel = pd.read_csv("26__weekday_out.csv")
cnm = chel.iloc[0:2880,1]
ssreg = np.sum((cnm-r1.a)**2)
ssreg


# In[ ]:



