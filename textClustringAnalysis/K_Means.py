import numpy
import random
import numpy.matlib

from textClustringAnalysis.common import log


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
    """随机初始化聚类中心"""
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
def k_Means(data, k, dist=dist_eld, maxIter=50, createCent=randCent):
    """k-means聚类算法"""
    Cent = createCent(data, k, dist=dist)
    m = data.shape[0]
    clusterChanged = True
    clusterLabel = [0] * m  # 簇下标
    iter = 0
    clusterLabel_map = {}
    while iter < maxIter and clusterChanged:
        clusterChanged = False
        iter += 1
        clusterLabel_map.clear()
        # 分配结点
        for i in range(m):
            minDist = dist(data[i, :], Cent[0, :])
            minIndex = 0
            for j in range(1, k):
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
    data = numpy.loadtxt('./feature/data_test', delimiter=",")
    data = numpy.mat(data)
    clusterLabel = k_Means(data, k=100, dist=dist_eld, createCent=randCent_plus)
    print(clusterLabel)
