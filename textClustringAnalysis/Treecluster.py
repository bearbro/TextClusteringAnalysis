from Bio.Cluster import treecluster
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn import mixture
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score
import matplotlib.pyplot as plt

from textClustringAnalysis.feature.common import myTFIDF, dict2Array
from textClustringAnalysis.feature.main import TC_PCA, TC, PCA
from textClustringAnalysis.main import size_of_cluster
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
import pandas as pd
import numpy as np

def gaussianMix(X, k):
    # 高斯聚类
    gmm = mixture.GMM(n_components=k)
    gmm.fit(X)
    pred_gmm = gmm.predict(X)
    print('gmm:', np.unique(pred_gmm))
    print('gmm:', silhouette_score(X, pred_gmm, metric='euclidean'))


def do_treecluster_images():
    """特征维度对各层次聚类的影响"""
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'
    txt_dict = getWordCount(outDir)

    xx = range(100, 1000, 100)
    xx = [100]
    for topN in xx:
        data, textNames = PCA(txt_dict, topN=topN)[:2]
        # method 's': 最小距离法  'm': 最大距离法 'c': 重心法  'a': 类平均法
        # dist e 欧式距离 u 余弦距离
        tree = treecluster(data=data, method='m', dist='e')
        tree2 = treecluster(data=data, method='s', dist='e')
        tree3 = treecluster(data=data, method='a', dist='e')
        tree4 = treecluster(data=data, method='c', dist='e')
        args = range(2, 50)
        args = [5, 8, 15]
        d = [[], [], [], [], []]  # 轮廓系数
        ksize = [[], [], [], [], []]  #最大类的大小
        for k in args:
            clusterid = tree.cut(nclusters=k)
            d[0].append(silhouette_score(data, clusterid, metric='euclidean'))
            ksize[0].append(max(size_of_cluster(clusterid)))
            clustering = AgglomerativeClustering(linkage='ward', n_clusters=k)  # ['ward','complete','average']
            clustering.fit(data)
            d[1].append(silhouette_score(data, clustering.labels_, metric='euclidean'))
            ksize[2].append(max(size_of_cluster(clustering.labels_)))
            clusterid2 = tree2.cut(nclusters=k)
            d[2].append(silhouette_score(data, clusterid2, metric='euclidean'))
            ksize[2].append(max(size_of_cluster(clusterid2)))
            clusterid3 = tree3.cut(nclusters=k)
            d[3].append(silhouette_score(data, clusterid3, metric='euclidean'))
            ksize[3].append(max(size_of_cluster(clusterid3)))
            clusterid4 = tree4.cut(nclusters=k)
            d[4].append(silhouette_score(data, clusterid4, metric='euclidean'))
            ksize[4].append(max(size_of_cluster(clusterid4)))
            # d[2].append(hierarchical(data, k, 'complete'))#m,e
            # d[3].append(hierarchical(data, k, 'average'))#a,e
        # 用subplot()方法绘制多幅图形
        plt.figure(figsize=(6, 6))
        # 创建第一个画板
        plt.figure(1)
        # 将第一个画板划分为2行1列组成的区块，并获取到第一块区域
        ax1 = plt.subplot(211)

        # 在第一个子区域中绘图
        for di in d:
            plt.plot(args, di, marker='o')
        # plt.legend(xx)
        plt.legend(range(len(d)))
        plt.xlabel = 'k'
        plt.ylabel = 'silhouette'
        plt.ylim(-1, 1)
        # plt.title('feature number=%d by PCA' % topN)

        # 选中第二个子区域，并绘图
        ax2 = plt.subplot(212)
        for di in ksize:
            plt.plot(args, di, marker='o')
        plt.legend(range(len(ksize)))
        plt.xlabel = 'k'
        plt.ylabel = 'MAXcluster'
        plt.ylim(0, 2000)
        ax1.set_title('feature number=%d by PCA' % topN)
        ax2.set_title("最大簇的元素个数")
        plt.savefig('./treecluster_images/feature number=%d by PCA' % topN)
        plt.show()


if __name__ == '__main__':
    # 层次聚类
    # tree = treecluster(data=data, method='m', dist='e')
    # for k in range(2,50):
    #     # y_pre = tree.cut(nclusters=k)
    #     clustering = AgglomerativeClustering(linkage='ward', n_clusters=k)  # ['ward','single'，'complete','average']
    #     clustering.fit(data)
    #     y_pre=clustering.labels_
    #     # print('%f silhouette[-1,1] 1==best' % silhouette_score(data, y_pre, metric='euclidean'))
    #     # print('%f DB 0==best' % davies_bouldin_score(data, y_pre))
    #     # print('%f Calinski-Harabaz bigger==batter' %   calinski_harabaz_score(data, y_pre))
    do_treecluster_images()

    # # 使用scipy.cluster进行层次聚类
    # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'
    # txt_dict = getWordCount(outDir)
    #
    # # topN = 1000
    # tfidf_dict = myTFIDF(txt_dict, itc=False)
    # # data, textNames, wordName = dict2Array(tfidf_dict)
    # data, textNames = PCA(txt_dict, topN=300)[:2]
    #
    # # 设置特征的名称
    # variables = range(data.shape[1])
    # # 设置编号
    # labels = textNames
    # # 通过pandas将数组转换成一个DataFrame
    # df = pd.DataFrame(data, columns=variables, index=labels)
    #
    # from scipy.spatial.distance import pdist, squareform
    #
    # # 获取距离矩阵
    # '''
    # pdist:计算两两样本间的欧式距离,返回的是一个一维数组
    # squareform：将数组转成一个对称矩阵
    # '''
    # dist_matrix = pd.DataFrame(squareform(pdist(df, metric="euclidean")),
    #                            columns=labels, index=labels)
    # # print(dist_matrix)
    # from scipy.cluster.hierarchy import linkage
    #
    # # 以全连接作为距离判断标准，获取一个关联矩阵
    # row_clusters = linkage(dist_matrix.values, method="complete", metric="euclidean")
    # # 将关联矩阵转换成为一个DataFrame
    # # 第一列表的是簇的编号，第二列和第三列表示的是簇中最不相似(距离最远)的编号，第四列表示的是样本的欧式距离，最后一列表示的是簇中样本的数量
    # clusters = pd.DataFrame(row_clusters, columns=["label 1", "label 2", "distance", "sample size"],
    #                         index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])])
    #
    # # 使用scipy的dendrogram来绘制树状图
    # from scipy.cluster.hierarchy import dendrogram
    # import matplotlib.pyplot as plt
    #
    # row_dendr = dendrogram(row_clusters, labels=labels, orientation='right')
    # plt.tight_layout()
    # plt.ylabel("欧式距离")
    # plt.show()
    # # 划分类簇
    # k = 8
    # y_pre = fcluster(row_clusters, criterion='maxclust', t=k)
    # kn = size_of_cluster(y_pre)
    # print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), np.mean(kn), kn)


## 当前最优结果 k=7 or 8 错的
#  数据集 /Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess
#  PCA topN=300 treecluster(data=data, method='a', dist='e') 达到0.4

# DB指数 零是最低分。接近零的值表示更好的分区。
# sklearn.metrics.davies_bouldin_score

# Calinski-Harabaz指数 - 也称为方差比标准 - 来评估模型
# sklearn.metrics.calinski_harabaz_score

# Dunn指数
