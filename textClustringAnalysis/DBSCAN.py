from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from textClustringAnalysis.feature.main import TC_PCA
from textClustringAnalysis.main import show_wbpj
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about2'
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'
    txt_dict = getWordCount(outDir)
    data, textNames = TC_PCA(txt_dict, topN=1600)[:2]


    # DBSCAN
    # 当空间聚类的密度不均匀、聚类间距差相差很大时，聚类质量较差，因为这种情况下参数MinPts和Eps选取困难。可以考虑KNA-DBSCAN（《浅析DBSCAN算法中参数设置问题的研究》）
    args = [i * 0.1 for i in range(1, 20)]
    d = [[], []]
    for eps in args:
        dbsc = DBSCAN(eps=eps,  # 邻域半径#todo DBSCAN 如何确实密度半径eps 和 最小密度点min_samples
                      min_samples=5,  # 最小样本点数，MinPts
                      metric='euclidean',
                      metric_params=None,
                      algorithm='auto',  # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
                      leaf_size=30,  # balltree,cdtree的参数
                      p=None,  #
                      n_jobs=1)
        dbsc.fit(data)
        labels = dbsc.labels_  # 聚类得到每个点的聚类标签 -1表示噪点
        # 去除噪声点
        y_pre = [labels[i] for i in range(len(labels)) if labels[i] != -1]
        d[0].append(len(y_pre))
        if len(y_pre) > 0:
            d[1].append(max(y_pre))
        else:
            d[1].append(-1)

    for di in d:
        plt.plot(args, di, marker='o')
        # plt.legend([])
        plt.show()
    ntextNames = [textNames[i] for i in range(len(labels)) if labels[i] != -1]
    ndata = data[[i for i in range(len(labels)) if labels[i] != -1], :]
    # 轮廓系数：si值介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中，
    # 越接近于0，表示样本应该在边界上。所有样本的si均值被称为聚类结果的轮廓系数。
    silhouette_s = metrics.silhouette_score(ndata, y_pre, metric='euclidean')
    show_wbpj(y_pre, ntextNames)
