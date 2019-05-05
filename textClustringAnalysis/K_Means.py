import numpy
import random
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics  # 导入sklearn效果评估模块
from textClustringAnalysis.feature.main import TC_PCA
from textClustringAnalysis.common import log
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
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    data = TC_PCA(txt_dict, topN=1800)[0]
    # clusterLabel = k_Means(data, k=10, dist=dist_eld, createCent=randCent_plus, elkan=True)
    # print(clusterLabel)

    # 肘方法看k值
    kList = list(range(10, 60, 1))

    d = []
    for i in kList:  # k取值1~11，做kmeans聚类，看不同k值对应的簇内误差平方和
        di = 0
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(data)
        # 成本函数
        # di += km.inertia_  # inertia簇内误差平方和
        # di += sum(numpy.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
        # #平均轮廓系数 #最高值为1，最差值为-1,0附近的值表示重叠的聚类
        y_pre = km.predict(data)
        di += metrics.silhouette_score(data, y_pre, metric='euclidean')
        d.append(di)

    plt.plot(kList, d, marker='o')
    plt.xlabel('number of clusters')
    plt.ylabel('distortions')
    # plt.ylim(0, 2000)
    plt.show()
    # a = [d[i] - d[i - 1] for i in range(1, len(d))]
    # plt.plot(kList[1:], a)
    # plt.show()
    #
    # k = kList[numpy.argmin(a) + 1]
    # # k = 29
    # # 训练聚类模型
    #
    # model_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)  # 建立模型对象
    # model_kmeans.fit(data)  # 训练聚类模型
    # y_pre = model_kmeans.predict(data)  # 预测聚类模型
    #
    # # https://www.cnblogs.com/niniya/p/8784947.html
