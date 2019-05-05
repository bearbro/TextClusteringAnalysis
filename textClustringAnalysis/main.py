from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from textClustringAnalysis.K_Means import k_Means, randCent_plus
from textClustringAnalysis.feature.main import TC
from textClustringAnalysis.feature.main import TC_PCA
from textClustringAnalysis.feature.main import PCA
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.preprocessor.start import preprocessor_main
from textClustringAnalysis.tag.getTag import tag2id, getTags


def test_findfeatureN():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    topN = 1800
    # newData_mat, txtName, wordName = TC(txt_dict, topN)
    # newData_mat2, txtName2 = PCA(txt_dict, topN=topN)
    # newData_mat3, txtName3 = TC_PCA(txt_dict, minTC=0, topN=topN)
    k = 29
    # 训练聚类模型
    Nlist = list(range(1, 50, 1))
    d = []
    featureFunction = [TC_PCA]
    for function in featureFunction:
        for topN in Nlist:
            data = function(txt_dict, topN=topN)[0]
            model_kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)  # 建立模型对象
            model_kmeans.fit(data)  # 训练聚类模型
            y_pre = model_kmeans.predict(data)  # 预测聚类模型
            # y_pre = k_Means(data, k, maxIter=300,createCent=randCent_plus, elkan=True)
            di = metrics.silhouette_score(data, y_pre, metric='euclidean')
            d.append(di)
        plt.plot(Nlist, d, marker='o')
        plt.xlabel('number of feature')
        plt.ylabel('silhouette_score_%s' % function.__name__)
        plt.show()


if __name__ == '__main__':
    dirName = '/Users/brobear/OneDrive/data-whitepaper/data/all_txt'
    # 预处理
    outDir = preprocessor_main(dirName, cache=True, q1=None, q2=None,
                               outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt')
    # 获得词频
    txt_dict = getWordCount(outDir)
    # 特征降维度
    topN = 1800
    data, textNames = TC_PCA(txt_dict, topN=topN)
    # 聚类
    k = 29
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    km.fit(data)
    y_pre = km.predict(data)
    # 获取标签
    y_true_dict = getTags(textNames)
    y_true = tag2id(textNames, y_true_dict)
    y_true = [i[0] for i in y_true]
    # 评价聚类结果
    # #评价指标
    # 样本距离最近的聚类中心的距离总和
    inertias = km.inertia_
    print("%f\t样本距离最近的聚类中心的距离总和" % inertias)
    # 调整后的兰德指数，范围是[-1,1]，值越大意味着聚类结果与真实情况越吻合。
    adjusted_rand_s = metrics.adjusted_rand_score(y_true, y_pre)
    print('%f\t调整后的兰德指数，范围是[-1,1]，值越大意味着聚类结果与真实情况越吻合' % adjusted_rand_s)
    # 互信息衡量聚类结果与实际类别的吻合程度，范围[0，1]，越接近1越吻合
    mutual_info_s = metrics.mutual_info_score(y_true, y_pre)
    print('%f\t互信息衡量聚类结果与实际类别的吻合程度，范围[0，1]，越接近1越吻合' % mutual_info_s)
    # 调整后的互信息，范围[-1，1]，越接近1越吻合
    adjusted_mutual_info_s = metrics.adjusted_mutual_info_score(y_true, y_pre)
    print('%f\t调整后的互信息，范围[-1，1]，越接近1越吻合' % adjusted_mutual_info_s)
    # 同质化得分（均一性）指每个簇中只包含单个类别的样本。
    # 如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C | K)，值越大则均一性越小。
    homogeneity_s = metrics.homogeneity_score(y_true, y_pre)
    print('%f\t如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C | K)，值越大则均一性越小。' % homogeneity_s)
    # 完整性（Completeness）指同类别样本被归类到相同的簇中。
    # 如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K | C)，值越大则完整性越小。
    completeness_s = metrics.completeness_score(y_true, y_pre)
    print('%f\t如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K | C)，值越大则完整性越小。' % completeness_s)
    # 单独考虑均一性或完整性都是片面的，因此引入两个指标的加权平均V - measure
    v_measure_s = metrics.v_measure_score(y_true, y_pre)
    print('%f\t均一性和完整性的的加权平均V - measure' % v_measure_s)
    # 轮廓系数：si值介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中，
    # 越接近于0，表示样本应该在边界上。所有样本的si均值被称为聚类结果的轮廓系数。
    silhouette_s = metrics.silhouette_score(data, y_pre, metric='euclidean')
    print('%f\t轮廓系数：介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中,0表示样本应该在边界上' % silhouette_s)
    # calinski&harabaz得分，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski - Harabasz分数会高
    calinski_harabaz_s = metrics.calinski_harabaz_score(data, y_pre)
    print('%f\t类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高' % calinski_harabaz_s)

# #评价指标
# inertias=model_kmeans.inertia_         #样本距离最近的聚类中心的距离总和
# adjusted_rand_s=metrics.adjusted_rand_score(y_true,y_pre)   #调整后的兰德指数
# mutual_info_s=metrics.mutual_info_score(y_true,y_pre)       #互信息
# adjusted_mutual_info_s=metrics.adjusted_mutual_info_score (y_true,y_pre)  #调整后的互信息
# homogeneity_s=metrics.homogeneity_score(y_true,y_pre)   #同质化得分
# completeness_s=metrics.completeness_score(y_true,y_pre)   #完整性得分
# v_measure_s=metrics.v_measure_score(y_true,y_pre)   #V-measure得分
# silhouette_s=metrics.silhouette_score(x,y_pre,metric='euclidean')   #轮廓系数
# calinski_harabaz_s=metrics.calinski_harabaz_score(x,y_pre)   #calinski&harabaz得分
#
#
# https://www.cnblogs.com/niniya/p/8784947.html
#
