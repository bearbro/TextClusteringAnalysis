import random

import matplotlib.pyplot as plt
import numpy
import numpy.matlib
from Bio.Cluster import distancematrix
from Bio.Cluster import kcluster
from Bio.Cluster import kmedoids
from numpy import mean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from textClustringAnalysis.common import log
from textClustringAnalysis.feature.common import myTFIDF, dict2Array
from textClustringAnalysis.feature.main import PCA
from textClustringAnalysis.main import size_of_cluster
from textClustringAnalysis.preprocessor.dataInfo import getWordCount


def sim_Cos(d1, d2, norm=True):
    """余弦相似度
    :type norm: 是否进行归一化
    """
    if d1.shape[0] == 1:
        cos = (d1 * d2.T)
    else:
        cos = (d1.T * d2)
    fm = (numpy.linalg.norm(d1) * numpy.linalg.norm(d2))
    if fm == 0:
        cos = 0
    else:
        cos /= fm
    if norm:  # 归一化
        cos = 0.5 * (1 + cos)
    return cos


def dist_cos(d1, d2):
    """余弦距离"""
    cos = sim_Cos(d1, d2, norm=True)
    dist = 1 - cos
    return dist


def dist_eld(d1, d2):
    """欧几里得距离"""
    dist = numpy.linalg.norm(d1 - d2)
    return dist


@log("useTime")
def randCent(data, k, dist):
    """随机初始化聚类中心"""
    n = data.shape[0]
    centroids = random.sample(range(n), k)  # 选k个
    centroids.sort()
    cent = data[centroids, :]
    return cent


@log("useTime")
def randCent_plus(data, k, dist=dist_eld):
    """k-means++ 初始化聚类中心"""
    n = data.shape[0]
    # 距离矩阵
    dists = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dists[i, j] = dist(data[i, :], data[j, :])
            dists[j, i] = dists[i, j]

    # 待选下标集
    indx = list(range(n))
    # 已经选择的下标集
    centroids = []
    # 选第一个
    centroid = random.sample(indx, 1)
    # 选中概率
    p = numpy.zeros(len(indx))
    p[centroid] = 1
    while len(centroids) < k:
        centroid = numpy.random.choice(a=indx, size=1, replace=False, p=p)
        centroids.append(centroid)
        indx.remove(centroid)
        p = p[:-1]
        for i in range(len(indx)):
            p[i] = dists[indx[i], centroids[0]]
            for j in range(1, len(centroids)):
                dij = dists[indx[i], centroids[j]]
                if dij < p[i]:
                    p[i] = dij
        # 归一化
        sumP = numpy.sum(p)
        p = p / sumP
    centroids.sort()
    cent = data[centroids, :]
    return cent


@log("useTime")
def k_Means(data, k, dist=dist_eld, maxIter=300, createCent=randCent_plus, elkan=True):
    """k-means聚类算法"""
    # 初始化聚类中心
    Cent = createCent(data, k, dist=dist)
    m = data.shape[0]
    clusterChanged = True
    clusterLabel = [0] * m  # 簇下标
    iter = 0
    clusterLabel_map = {}
    if elkan:  # elkan k-means聚类算法,利用a+b>=c减少不必要的距离的计算
        dists_k = numpy.zeros((k, k))  # 计算各簇中心间的距离
    while iter < maxIter and clusterChanged:
        clusterChanged = False
        iter += 1
        clusterLabel_map.clear()
        if elkan:
            # 计算各簇中心间的距离
            for i in range(k):
                for j in range(i + 1, k):
                    dists_k[i, j] = dist(Cent[i, :], Cent[j, :])
                    dists_k[j, i] = dists_k[i, j]
        # 分配结点
        for i in range(m):
            minDist = dist(data[i, :], Cent[0, :])
            minIndex = 0
            for j in range(1, k):
                if elkan:
                    # 规则1优化 2*D_xj1 <= D_j1j2 则 D_xj1<=D_xj2
                    if minDist * 2 <= dists_k[minIndex, j]:
                        continue
                    # 规则2优化 D(x,j2)≥max{0,D(x,j1)−D(j1,j2)} # todo
                    if minDist <= max(0, minDist - dists_k[minIndex, j]):
                        continue
                distJI = dist(data[i, :], Cent[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterLabel[i] != minIndex:
                clusterChanged = True
                clusterLabel[i] = minIndex
            if minIndex in clusterLabel_map:
                clusterLabel_map[minIndex].append(i)
            else:
                clusterLabel_map[minIndex] = [i]
        # 更新中心
        for i in range(k):
            Cent[i, :] = numpy.mean(data[clusterLabel_map[i], :], axis=0)
    print(iter)
    return clusterLabel


if __name__ == '__main__':

    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'
    txt_dict = getWordCount(outDir)
    tfidf_dict = myTFIDF(txt_dict, itc=False)
    data, textNames, wordName = dict2Array(tfidf_dict)
    # 降维
    topN = 1200
    data, textNames = PCA(txt_dict, topN=topN, itc=False)[:2]
    # 确定特征维数
    for x in [i * 0.1 for i in range(1, 10)]:
        data, textNames = PCA(txt_dict, topN=x, itc=False)[:2]
        print(x, data.shape)
    # 结果：0.1 74 0.2 204 0.3 357 0.4 519 0.5 684 0.6 851 0.7 1022 0.8 1198 0.9 1387
    # [74, 204, 357, 519, 684, 851, 1022, 1198, 1387]
    #
    #
    # # 肘方法看k值
    # kList = range(5, 40, 1)
    # d = []
    # for k in kList:
    #     km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=9000, random_state=0, n_jobs=2)
    #     km.fit(data)
    #     y_pre = km.predict(data)
    #     # 成本函数
    #     # di = km.inertia_  # inertia簇内误差平方和
    #     # di = sum(numpy.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
    #     di = silhouette_score(data, y_pre, metric='euclidean') #平均轮廓系数 #最高值为1，最差值为-1,0附近的值表示重叠的聚类
    #     kn = size_of_cluster(y_pre)
    #     print(k, di, max(kn), numpy.mean(kn), kn)
    #     d.append(di)
    #
    # plt.plot(kList, d, marker='o')
    # plt.xlabel('number of clusters')
    # plt.ylabel('distortions')
    # plt.show()

    # # k_Means聚类
    # k = 13
    # # y_pre = k_Means(data, k=k, dist=dist_eld, createCent=randCent_plus, elkan=True)#手写的
    # km = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=9000)
    # km.fit(data)
    # y_pre = km.predict(data)
    # print('轮廓系数 %f' % silhouette_score(data, y_pre, metric='euclidean'))
    # # 统计各类的规模
    # kn = size_of_cluster(y_pre)
    # print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), mean(kn), kn)
    #
    # # 使用Bio.Cluster库进行聚类 余弦距离 k中心
    # d = [[], [], [],
    #      [], [], [],
    #      [], [], []]
    # x = list(range(2, 15, 1)) + list(range(12, 25, 4)) + [29, 40]
    # for k in x:
    #     # kmeans
    #     # dist： e 欧几里得距离 u余弦距离
    #     # method：a 均值 m 中间值 确定中心点
    #     #
    #     # kmeans-e欧几里得距离
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='e', npass=20, method='a')
    #     # metric： euclidean 欧几里得  cosine 余弦距离
    #     d[0].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     # kmeans-u余弦距离
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='u', npass=20, method='a')
    #     d[1].append(silhouette_score(data, clusterid, metric='cosine'))
    #     d[2].append(silhouette_score(data, clusterid, metric='euclidean'))  # 好
    #     #
    #     # kmedians-e欧几里得距离
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='e', npass=20, method='m')
    #     # metric： euclidean 欧几里得  cosine 余弦距离
    #     d[3].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     # kmedians-u余弦距离
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='u', npass=20, method='m')
    #     d[4].append(silhouette_score(data, clusterid, metric='cosine'))
    #     d[5].append(silhouette_score(data, clusterid, metric='euclidean'))
    #
    #     # kmedoids-e
    #     # 计算距离矩阵
    #     dist_data = distancematrix(data, dist='e')
    #     clusterid, error, nfound = kmedoids(distance=dist_data, nclusters=k, npass=20)
    #     # metric euclidean 欧几里得  cosine 余弦距离
    #     d[6].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     # kmedoids-u
    #     dist_data = distancematrix(data, dist='u')
    #     clusterid, error, nfound = kmedoids(distance=dist_data, nclusters=k, npass=20)
    #     # metric euclidean 欧几里得  cosine 余弦距离
    #     d[7].append(silhouette_score(data, clusterid, metric='cosine'))
    #     d[8].append(silhouette_score(data, clusterid, metric='euclidean'))
    #
    # for di in d:
    #     if len(di) > 1:
    #         plt.plot(x, di, marker='o')
    # plt.legend(['kmeans-e', 'kmeans-u-u', 'kmeans-u-e',
    #             'kmedians-e', 'kmedians-u-u', 'kmedians-u-e',
    #             'kmedoids-e', 'kmedoids-u-u', 'kmedoids-u-e'])
    # plt.show()

    #
    # # https://www.cnblogs.com/niniya/p/8784947.html
