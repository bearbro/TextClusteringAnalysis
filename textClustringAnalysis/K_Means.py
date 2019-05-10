import numpy
import random
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics  # 导入sklearn效果评估模块

from textClustringAnalysis.feature.common import myTFIDF, dict2Array
from textClustringAnalysis.feature.main import TC_PCA, TC, PCA
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
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from Bio.Cluster import kcluster
    from Bio.Cluster import distancematrix
    import matplotlib.pyplot as plt
    from Bio.Cluster import kmedoids

    import numpy as np

    txt_dict = getWordCount('/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess')
    data = PCA(txt_dict, topN=300)[0]
    # topN = 1000
    # tfidf_dict = myTFIDF(txt_dict, itc=False)
    # data, textNames, wordName = dict2Array(tfidf_dict)
    # k=8
    for k in range(2, 30):
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=3000, random_state=0, n_jobs=4)
        km.fit(data)
        y_pre = km.predict(data)
        # clusterid, error, nfound = kcluster(data, nclusters=k, dist='e', npass=20, method='a')
        kn = [0] * (max(y_pre) + 1)
        for i in y_pre:
            kn[i] += 1
        print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), np.mean(kn), kn)
    # clusterLabel = k_Means(data, k=10, dist=dist_eld, createCent=randCent_plus, elkan=True)
    # print(clusterLabel)
    #
    # d = [[], [], [], [], [], [], []]
    # x = list(range(2, 15,1))#+list(range(12, 25,4))+[ 29, 40]
    # for k in x:
    #     # kmeans
    #     # dist： e 欧几里得距离 u余弦距离
    #     # method：a 均值 m 中间值 确定中心点
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='e', npass=20, method='a')
    #     # metric： euclidean 欧几里得  cosine 余弦距离
    #     d[0].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     # kmeans-u余弦距离
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='u', npass=20, method='a')
    #     d[3].append(silhouette_score(data, clusterid, metric='cosine'))
    #     d[4].append(silhouette_score(data, clusterid, metric='euclidean'))  # 好
    #     #
    #     # k-medians
    #     clusterid, error, nfound = kcluster(data, nclusters=k, dist='e', npass=20, method='m')
    #     # metric： euclidean 欧几里得  cosine 余弦距离
    #     d[1].append(silhouette_score(data, clusterid, metric='euclidean'))
    #
    #     # kmedoids
    #     # 计算距离矩阵
    #     dist_data = distancematrix(data, dist='e')
    #     clusterid, error, nfound = kmedoids(distance=dist_data, nclusters=k, npass=20)
    #     # metric euclidean 欧几里得  cosine 余弦距离
    #     d[2].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     # kmedoids-u
    #     dist_data = distancematrix(data, dist='u')
    #     clusterid, error, nfound = kmedoids(distance=dist_data, nclusters=k, npass=20)
    #     # metric euclidean 欧几里得  cosine 余弦距离
    #     d[5].append(silhouette_score(data, clusterid, metric='cosine'))
    #     d[6].append(silhouette_score(data, clusterid, metric='euclidean'))
    #
    # for di in d:
    #     plt.plot(x, di, marker='o')
    # plt.legend(['kmeans-e', 'kmedians-e', 'kmedoids-e', 'kmeans-u', 'kmeans-u-e', 'kmedoids-u', 'kmedoids-u-e'])
    # plt.show()

    # 肘方法看k值
    # kList = range(5, 40, 1)
    # # kList = x
    # d = []
    # for i in kList:
    #     di = 0
    #     km = KMeans(n_clusters=i, init='k-means++', n_init=20, max_iter=3000, random_state=0,n_jobs=4)
    #     km.fit(data)
    #     # 成本函数
    #     # di += km.inertia_  # inertia簇内误差平方和
    #     # di += sum(numpy.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
    #     # #平均轮廓系数 #最高值为1，最差值为-1,0附近的值表示重叠的聚类
    #     y_pre = km.predict(data)
    #     di += silhouette_score(data, y_pre, metric='euclidean')
    #     kn = [0] * (max(y_pre) + 1)
    #     for i in y_pre:
    #         kn[i] += 1
    #     print(i, silhouette_score(data, y_pre, metric='euclidean'), max(kn), numpy.mean(kn), kn)
    #     d.append(di)
    #
    # plt.plot(kList, d, marker='o')
    # plt.xlabel('number of clusters')
    # plt.ylabel('distortions')
    # # plt.ylim(0, 2000)
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
'''
PyDev console: starting.
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
runfile('/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis/K_Means.py', wdir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis')
GetWordCount_useTime getWordCount 5431.86092376709 ms
useTime myTFIDF_dict 2591.348886489868 ms
useTime myTFIDF 2591.4080142974854 ms
useTime dict2Array 3968.7139987945557 ms
8 0.003236440063488912 1425 373.4 [251, 1425, 48, 22, 121]
8 0.0037034426260885308 1316 311.1666666666667 [22, 155, 152, 105, 1316, 117]
8 0.0040353835829753494 1570 266.7142857142857 [18, 22, 1570, 92, 23, 59, 83]
8 0.0047025113634818864 1311 233.375 [70, 156, 22, 133, 51, 1311, 91, 33]
8 0.005136675979023439 1044 207.44444444444446 [137, 162, 41, 193, 1044, 22, 52, 95, 121]
8 0.004931831927767722 937 186.7 [136, 154, 41, 214, 937, 22, 52, 93, 122, 96]
8 0.004495147112567703 1159 169.72727272727272 [62, 80, 1159, 22, 35, 65, 90, 61, 83, 158, 52]
8 0.0048536525634321395 929 155.58333333333334 [72, 98, 33, 122, 224, 22, 51, 76, 89, 73, 929, 78]
8 0.004822977304247962 1021 143.6153846153846 [57, 76, 1021, 22, 34, 62, 84, 60, 79, 113, 52, 126, 81]
8 0.004635901774976122 869 133.35714285714286 [28, 96, 33, 107, 198, 22, 51, 75, 83, 71, 869, 6, 102, 126]
8 0.0049760757628634555 828 124.46666666666667 [28, 94, 33, 103, 183, 22, 51, 65, 82, 70, 828, 6, 102, 119, 81]
8 0.00509514797470572 669 116.6875 [46, 136, 259, 105, 51, 669, 102, 33, 51, 46, 28, 41, 204, 7, 67, 22]
8 0.005064538305221128 665 109.82352941176471 [46, 136, 257, 105, 50, 665, 102, 33, 51, 46, 28, 40, 204, 7, 67, 22, 8]
8 0.005064726601673844 660 103.72222222222223 [46, 135, 255, 104, 50, 660, 103, 33, 51, 46, 28, 40, 196, 7, 66, 22, 8, 17]
8 0.005815643456769889 543 98.26315789473684 [17, 41, 80, 15, 31, 38, 211, 162, 101, 22, 2, 76, 42, 15, 41, 42, 543, 168, 220]
8 0.0054056198383116184 579 93.35 [23, 130, 269, 95, 50, 579, 98, 33, 50, 46, 30, 38, 127, 7, 59, 22, 7, 15, 67, 122]
8 0.0055495236650008025 657 88.9047619047619 [24, 44, 144, 22, 53, 22, 48, 33, 48, 47, 456, 93, 7, 48, 37, 3, 5, 26, 22, 28, 657]
8 0.005624603716360746 623 84.86363636363636 [23, 43, 139, 22, 53, 21, 48, 33, 48, 44, 443, 89, 7, 48, 37, 3, 5, 26, 21, 28, 623, 63]
8 0.0058240442020192065 558 81.17391304347827 [65, 105, 22, 28, 52, 20, 51, 113, 5, 71, 28, 6, 312, 25, 42, 177, 13, 2, 59, 14, 63, 36, 558]
8 0.005502645373149283 390 77.79166666666667 [22, 37, 94, 22, 50, 14, 44, 32, 48, 41, 370, 91, 7, 45, 37, 3, 5, 13, 20, 26, 380, 56, 390, 20]
8 0.005541448235826162 388 74.68 [22, 37, 91, 22, 49, 14, 43, 32, 48, 40, 355, 91, 7, 45, 37, 3, 5, 13, 20, 26, 369, 56, 388, 20, 34]
8 0.005665119505158111 383 71.8076923076923 [22, 37, 91, 22, 48, 14, 43, 32, 48, 40, 352, 91, 7, 45, 37, 3, 5, 13, 20, 26, 365, 56, 383, 20, 34, 13]
8 0.005709228079414905 381 69.14814814814815 [22, 37, 87, 22, 47, 14, 43, 32, 48, 40, 348, 91, 7, 45, 37, 3, 5, 13, 20, 26, 363, 55, 381, 20, 34, 13, 14]
8 0.005802617290178534 380 66.67857142857143 [22, 37, 87, 22, 47, 14, 43, 32, 48, 38, 342, 90, 6, 45, 34, 3, 5, 13, 20, 26, 356, 51, 380, 19, 35, 13, 14, 25]
8 0.005904865605159299 382 64.37931034482759 [21, 37, 86, 22, 47, 12, 43, 32, 48, 38, 341, 83, 6, 45, 33, 3, 5, 13, 19, 26, 340, 49, 382, 18, 35, 13, 14, 25, 31]
8 0.005912510020657216 383 62.233333333333334 [21, 37, 84, 22, 47, 12, 43, 32, 48, 38, 336, 83, 6, 45, 33, 3, 5, 13, 19, 26, 339, 49, 383, 17, 35, 13, 14, 25, 31, 8]
8 0.005781936916727134 376 60.225806451612904 [21, 37, 84, 8, 45, 12, 41, 32, 48, 38, 335, 83, 6, 45, 33, 3, 5, 13, 19, 26, 346, 48, 376, 17, 35, 13, 14, 24, 31, 8, 21]
8 0.006415750840062313 609 58.34375 [30, 13, 21, 2, 50, 609, 80, 61, 21, 56, 262, 39, 8, 31, 20, 67, 49, 10, 20, 16, 19, 66, 21, 31, 57, 13, 29, 22, 72, 14, 40, 18]
8 0.006250188870551721 593 56.57575757575758 [30, 13, 21, 2, 48, 593, 79, 61, 21, 56, 254, 39, 8, 29, 20, 66, 49, 10, 20, 16, 23, 66, 21, 31, 57, 13, 29, 22, 71, 12, 40, 18, 29]
8 0.006291229316363154 557 54.911764705882355 [29, 13, 21, 2, 48, 557, 77, 59, 21, 56, 223, 39, 8, 29, 20, 66, 47, 10, 20, 16, 23, 67, 21, 31, 57, 13, 29, 22, 71, 12, 39, 18, 27, 76]
8 0.006306077702831018 558 53.34285714285714 [28, 13, 21, 2, 48, 558, 48, 58, 20, 56, 213, 38, 8, 28, 20, 66, 47, 10, 20, 15, 25, 67, 18, 30, 53, 13, 29, 22, 71, 12, 38, 18, 27, 75, 52]
8 0.005911037460334614 550 51.861111111111114 [28, 13, 21, 2, 48, 550, 48, 58, 20, 56, 216, 38, 8, 28, 20, 61, 47, 10, 20, 15, 25, 67, 18, 30, 53, 13, 29, 22, 71, 12, 38, 18, 27, 73, 52, 12]
8 0.0059732116070078745 542 50.45945945945946 [28, 13, 21, 2, 48, 542, 41, 57, 20, 56, 215, 38, 8, 27, 20, 60, 47, 10, 20, 15, 23, 65, 16, 30, 53, 13, 28, 22, 70, 12, 38, 18, 27, 70, 50, 12, 32]
8 0.0056841781978535354 529 49.13157894736842 [26, 12, 21, 2, 47, 529, 41, 57, 19, 56, 215, 33, 8, 27, 20, 60, 47, 10, 20, 15, 23, 49, 16, 30, 52, 12, 27, 22, 70, 11, 37, 18, 27, 69, 48, 12, 32, 47]


'''
