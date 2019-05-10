from Bio.Cluster import treecluster
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn import mixture
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score
import matplotlib.pyplot as plt

from textClustringAnalysis.feature.common import myTFIDF, dict2Array
from textClustringAnalysis.feature.main import TC_PCA, TC, PCA
from textClustringAnalysis.preprocessor.dataInfo import getWordCount


def gaussianMix(X, k):
    gmm = mixture.GMM(n_components=k)
    gmm.fit(X)
    pred_gmm = gmm.predict(X)
    print('gmm:', np.unique(pred_gmm))
    print('gmm:', silhouette_score(data, pred_gmm, metric='euclidean'))


def hierarchical(X, k, linkage='ward'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=k)  # ['ward','complete','average']
    clustering.fit(X)
    return silhouette_score(X, clustering.labels_, metric='euclidean')


def do_treecluster_images():
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess'
    txt_dict = getWordCount(outDir)

    xx = range(100, 1000, 100)
    for topN in xx:
        data, textNames = PCA(txt_dict, topN=topN)[:2]
        # method 's': 最小距离法  'm': 最大距离法 'c': 重心法  'a': 类平均法
        # dist e 欧式距离 u 余弦距离
        tree = treecluster(data=data, method='m', dist='e')
        tree2 = treecluster(data=data, method='s', dist='e')
        tree3 = treecluster(data=data, method='a', dist='e')
        tree4 = treecluster(data=data, method='c', dist='e')
        args = range(2, 50)
        d = [[], [], [], [], []]
        for k in args:
            clusterid = tree.cut(nclusters=k)
            d[0].append(silhouette_score(data, clusterid, metric='euclidean'))
            d[1].append(hierarchical(data, k, 'ward'))
            d[2].append(silhouette_score(data, tree2.cut(nclusters=k), metric='euclidean'))
            d[3].append(silhouette_score(data, tree3.cut(nclusters=k), metric='euclidean'))
            d[4].append(silhouette_score(data, tree4.cut(nclusters=k), metric='euclidean'))
            # d[2].append(hierarchical(data, k, 'complete'))#m,e
            # d[3].append(hierarchical(data, k, 'average'))#a,e
        for di in d:
            plt.plot(args, di, marker='o')
        # plt.legend(xx)
        plt.legend(range(len(d)))
        plt.xlabel = 'k'
        plt.ylabel = 'silhouette'
        plt.ylim(-1, 1)
        plt.title('feature number=%d by PCA' % topN)

        plt.savefig('./treecluster_images/feature number=%d by PCA' % topN)
        plt.show()


if __name__ == '__main__':
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess'
    txt_dict = getWordCount(outDir)

    # topN = 1000
    tfidf_dict = myTFIDF(txt_dict, itc=False)
    # data, textNames, wordName = dict2Array(tfidf_dict)
    data, textNames = PCA(txt_dict, topN=300)[:2]
    import pandas as pd
    import numpy as np

    # 设置特征的名称
    variables = range(data.shape[1])
    # 设置编号
    labels = textNames
    # 通过pandas将数组转换成一个DataFrame
    df = pd.DataFrame(data, columns=variables, index=labels)

    from scipy.spatial.distance import pdist, squareform

    # 获取距离矩阵
    '''
    pdist:计算两两样本间的欧式距离,返回的是一个一维数组
    squareform：将数组转成一个对称矩阵
    '''
    dist_matrix = pd.DataFrame(squareform(pdist(df, metric="euclidean")),
                               columns=labels, index=labels)
    # print(dist_matrix)
    from scipy.cluster.hierarchy import linkage

    # 以全连接作为距离判断标准，获取一个关联矩阵
    row_clusters = linkage(dist_matrix.values, method="complete", metric="euclidean")
    # 将关联矩阵转换成为一个DataFrame
    # 第一列表的是簇的编号，第二列和第三列表示的是簇中最不相似(距离最远)的编号，第四列表示的是样本的欧式距离，最后一列表示的是簇中样本的数量
    clusters = pd.DataFrame(row_clusters, columns=["label 1", "label 2", "distance", "sample size"],
                            index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])])
    # print(clusters)
    # 使用scipy的dendrogram来绘制树状图
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt

    row_dendr = dendrogram(row_clusters, labels=labels, orientation='right')
    plt.tight_layout()
    plt.ylabel("欧式距离")
    plt.show()
    k = 8
    y_pre = fcluster(row_clusters, criterion='maxclust', t=k)
    kn = [0] * (max(y_pre) + 1)
    for i in y_pre:
        kn[i] += 1
    print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), np.mean(kn), kn)
    # xx = range(100, 1000, 100)
    # for topN in xx:
    # data, textNames = PCA(txt_dict, topN=300)[:2]
    # # method 's': 最小距离法  'm': 最大距离法 'c': 重心法  'a': 类平均法
    # # dist e 欧式距离 u 余弦距离
    # tree = treecluster(data=data, method='a', dist='e')
    # # tree2 = treecluster(data=data, method='s', dist='e')
    # # tree3 = treecluster(data=data, method='a', dist='e')
    # # tree4 = treecluster(data=data, method='c', dist='e')
    # args = range(2, 50)
    # d = [[], [], []]
    # for k in args:
    #     clusterid = tree.cut(nclusters=k)
    #     d[0].append(silhouette_score(data, clusterid, metric='euclidean'))
    #     d[1].append(davies_bouldin_score(data, clusterid))
    #     d[2].append(calinski_harabaz_score(data, clusterid))
    #
    #     # d[1].append(hierarchical(data, k, 'ward'))
    #     # d[2].append(silhouette_score(data, tree2.cut(nclusters=k), metric='euclidean'))
    #     # d[3].append(silhouette_score(data, tree3.cut(nclusters=k), metric='euclidean'))
    #     # d[4].append(silhouette_score(data, tree4.cut(nclusters=k), metric='euclidean'))
    #     # d[2].append(hierarchical(data, k, 'complete'))#m,e
    #     # d[3].append(hierarchical(data, k, 'average'))#a,e
    # for di in d:
    #     plt.plot(args, di, marker='o')
    # # plt.legend(xx)
    # plt.legend(['silhouette', 'davies_bouldin', 'calinski_harabaz'])
    # plt.xlabel = 'k'
    # # plt.ylabel = 'silhouette'
    # # plt.ylim(-1, 1)
    # # plt.title('feature number=%d by TSNE' % topN)
    # #
    # # plt.savefig('./treecluster_images/feature number=%d by TSNE' % topN)
    # plt.show()

## 当前最优结果 k=7 or 8 错的
#  数据集 /Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess
#  PCA topN=300 treecluster(data=data, method='a', dist='e') 达到0.4

# DB指数 零是最低分。接近零的值表示更好的分区。
# sklearn.metrics.davies_bouldin_score

# Calinski-Harabaz指数 - 也称为方差比标准 - 来评估模型
# sklearn.metrics.calinski_harabaz_score

# Dunn指数
