import random

import numpy
from Bio.Cluster import treecluster
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score

from textClustringAnalysis.K_Means import k_Means, randCent_plus
from textClustringAnalysis.feature.common import dict2Array
from textClustringAnalysis.feature.main import TC
from textClustringAnalysis.feature.main import TC_PCA
from textClustringAnalysis.feature.main import PCA
from textClustringAnalysis.preprocessor.dataInfo import getWordCount, showDistplot
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


def wbpj(y_true, y_pre):
    # 获得外部评价
    # def analysis_restul(y_pre, y_true):
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
    # 如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C|K)，值越大则均一性越小。
    homogeneity_s = metrics.homogeneity_score(y_true, y_pre)
    print('%f\t如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C|K)，值越大则均一性越小。' % homogeneity_s)
    # 完整性（Completeness）指同类别样本被归类到相同的簇中。
    # 如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K|C)，值越大则完整性越小。
    completeness_s = metrics.completeness_score(y_true, y_pre)
    print('%f\t如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K|C)，值越大则完整性越小。' % completeness_s)
    # 单独考虑均一性或完整性都是片面的，因此引入两个指标的加权平均V - measure
    v_measure_s = metrics.v_measure_score(y_true, y_pre)
    print('%f\t均一性和完整性的的加权平均V - measure' % v_measure_s)

    # calinski&harabaz得分，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski - Harabasz分数会高
    calinski_harabaz_s = metrics.calinski_harabaz_score(data, y_pre)
    print('%f\t类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高' % calinski_harabaz_s)

    print('ARI\tMI\tAMI\thomo\tcomp\tv_m\tc&h\n%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d' %
          (adjusted_rand_s, mutual_info_s, adjusted_mutual_info_s, homogeneity_s,
           completeness_s, v_measure_s, calinski_harabaz_s))


def showpj(y_pre, textNames):
    # 获取标签
    y_true_dict = getTags(textNames)
    # 非类别标签
    deleteTag = ['Other', 'Platform', 'Cryptocurrency', 'Business services',
                 'Investment', 'Smart Contract', 'Software', 'Internet', 'Infrastructure', 'Entertainment']
    for v in y_true_dict.values():
        for i in deleteTag:
            if i in v:
                v.remove(i)
    y_pre2true = {}
    for i in range(len(textNames)):
        if y_pre[i] not in y_pre2true:
            y_pre2true[y_pre[i]] = []
        y_pre2true[y_pre[i]].append(y_true_dict[textNames[i]])
    y_pre2true_dict = {}
    for k, v in y_pre2true.items():
        y_pre2true_dict[k] = {}
        vv = [i for i in v if i != []]
        for i in v:
            for j in i:
                y_pre2true_dict[k][j] = y_pre2true_dict[k].get(j, 0.0) + 1 / len(i) / len(vv)
    y_pre2true_array, cid, tagList = dict2Array(y_pre2true_dict)
    colorListAll = ['yellowgreen', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
                    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
                    'chocolate',
                    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
                    'darkgoldenrod',
                    'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                    'darkorchid',
                    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
                    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite',
                    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
                    'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
                    'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
                    'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
                    'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
                    'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
                    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
                    'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
                    'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
                    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
                    'red',
                    'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
                    'sienna',
                    'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan',
                    'teal',
                    'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
                    'aliceblue']
    # colorList = random.sample(colorListAll, y_pre2true_array.shape[1])
    colorList = [colorListAll[i * 4] for i in range(y_pre2true_array.shape[1])]
    for i in range(y_pre2true_array.shape[1] - 1, -1, -1):
        plt.bar(cid, +y_pre2true_array[:, i], color=colorList[i], edgecolor='black', alpha=0.9)
    plt.legend(tagList[::-1], bbox_to_anchor=(0, 1), loc=3, ncol=3, borderaxespad=0)
    plt.show()
    y_pre2true_array2 = y_pre2true_array.copy()
    for i in range(1, y_pre2true_array2.shape[1]):
        y_pre2true_array2[:, i] += y_pre2true_array2[:, i - 1]
    for i in range(y_pre2true_array.shape[1] - 1, -1, -1):
        plt.bar(cid, +y_pre2true_array2[:, i], color=colorList[i], edgecolor='black', alpha=1)
    plt.legend(tagList[::-1], bbox_to_anchor=(0, 1), loc=3, ncol=3, borderaxespad=0)
    plt.show()


def main_K_means(featureFunction, topN, k, dirName=None, outDir=None, have_return=True, have_true=None):
    if dirName is not None:
        # 预处理
        outDir = preprocessor_main(dirName,
                                   cache=None,
                                   q1=None, q2=None,
                                   outDir=None)
    # 获得词频
    txt_dict = getWordCount(outDir)
    # 特征降维度
    data, textNames = featureFunction(txt_dict, topN=topN)[:2]
    # k_Means聚类
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=9000)
    km.fit(data)
    y_pre = km.predict(data)
    # 轮廓系数：si值介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中，
    # 越接近于0，表示样本应该在边界上。所有样本的si均值被称为聚类结果的轮廓系数。
    silhouette_s = metrics.silhouette_score(data, y_pre, metric='euclidean')
    if have_return:
        return silhouette_s
    else:
        showpj(y_pre, textNames)
        # 评价聚类结果
        # #评价指标
        # 样本距离最近的聚类中心的距离总和
        inertias = km.inertia_
        print("%f\t样本距离最近的聚类中心的距离总和" % inertias)
        print('%f\t轮廓系数：介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中,0表示样本应该在边界上' % silhouette_s)
        silhouette_list = metrics.silhouette_samples(data, y_pre, metric='euclidean')
        showDistplot(silhouette_list, title='样本轮廓系数分布')
        if have_true is not None:
            y_true = have_true
            wbpj(y_true, y_pre)


def show_tsne(data, y_pre):
    # 在平面上显示聚类结果
    tsne = TSNE(n_components=2)
    a = tsne.fit_transform(data)  # 进行数据降维,降成两维
    import matplotlib.pyplot as plt
    wk = wk = max(y_pre) + 1
    for i in range(wk):
        d = [j for j in range(a.shape[0]) if y_pre[j] == i]
        plt.scatter(a[d, 0], a[d, 1], s=2, alpha=1)
    plt.legend(range(wk))
    plt.show()
    # # 在三维空间显示聚类结果
    # tsne = TSNE(n_components=3)
    # a = tsne.fit_transform(data)  # 进行数据降维,降成三维
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax=Axes3D(fig)
    # wk = max(y_pre)+1 #or min(10, max(y_pre))
    # for i in range(wk):
    #     d = [j for j in range(a.shape[0]) if y_pre[j] == i]
    #     ax.scatter(a[d, 0], a[d, 1], a[d, 2],'.',alpha=1,s=2)
    # plt.legend(range(wk))
    # plt.show()


if __name__ == '__main__':
    from sklearn.manifold import TSNE
    import pandas as pd

    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess'
    txt_dict = getWordCount(outDir)
    topN = 1000
    data, textNames = PCA(txt_dict, topN=topN)[:2]
    k = 13
    # k_Means聚类
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=9000)
    km.fit(data)
    y_pre = km.predict(data)
    print('%f silhouette[-1,1] 1==best' % silhouette_score(data, y_pre, metric='euclidean'))
    # print('%f DB 0==best'%davies_bouldin_score(data, y_pre))

    # print('%f Calinski-Harabaz bigger==batter' %  calinski_harabaz_score(data, y_pre))
    # 层次聚类
    # tree = treecluster(data=data, method='m', dist='e')
    #
    # for k in range(2,50):
    #     # y_pre = tree.cut(nclusters=k)
    #     clustering = AgglomerativeClustering(linkage='ward', n_clusters=k)  # ['ward','single'，'complete','average']
    #     clustering.fit(data)
    #     y_pre=clustering.labels_
    #     # print('%f silhouette[-1,1] 1==best' % silhouette_score(data, y_pre, metric='euclidean'))
    #     # print('%f DB 0==best' % davies_bouldin_score(data, y_pre))
    #     # print('%f Calinski-Harabaz bigger==batter' %   calinski_harabaz_score(data, y_pre))
    #
    #     # show_tsne(data, y_pre)
    kn = [0] * (max(y_pre) + 1)
    for i in y_pre:
        kn[i] += 1
    print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), numpy.mean(kn), kn)

    #
    # d = []
    # argList = range(2, 30)
    # for arg in argList:
    #     di = main_K_means(TC, topN=600, k=arg, dirName=None,
    #                       outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about2', have_return=True,
    #                       have_true=None)
    #     d.append(di)
    # plt.plot(argList, d, marker='o')
    # plt.show()

    # # dirName = '/Users/brobear/OneDrive/data-whitepaper/data/all_txt'
    # # # 预处理
    # # outDir = preprocessor_main(dirName,
    # #                            cache='/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess2',
    # #                            q1=1000, q2=8000,
    # #                            outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2')
    # # 获得词频
    # # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt'
    # dirName = '/Users/brobear/OneDrive/data-whitepaper/data/txt_about_all'
    # # 预处理
    # outDir = preprocessor_main(dirName,
    #                            cache='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about_preproccess2',
    #                            q1=50, q2=200,
    #                            outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about2')
    # # # 获得词频
    # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about2'
    # txt_dict = getWordCount(outDir)
    # # 特征降维度
    # topN = 300
    # data, textNames = TC_PCA(txt_dict, topN=topN)
    # # k_Means聚类
    # k = 17
    # km = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=9000)
    # km.fit(data)
    # y_pre = km.predict(data)
    # # 密度聚类
    # # todo
    #
    # # 获取标签
    # y_true_dict = getTags(textNames)
    # # 非类别标签
    # deleteTag = ['Other', 'Platform', 'Cryptocurrency', 'Business services',
    #              'Investment', 'Smart Contract', 'Software', 'Internet', 'Infrastructure', 'Entertainment']
    # for v in y_true_dict.values():
    #     for i in deleteTag:
    #         if i in v:
    #             v.remove(i)
    # y_true = tag2id(textNames, y_true_dict)
    # y_pre2true = {}
    # for i in range(len(textNames)):
    #     if y_pre[i] not in y_pre2true:
    #         y_pre2true[y_pre[i]] = []
    #     y_pre2true[y_pre[i]].append(y_true_dict[textNames[i]])
    # y_pre2true_dict = {}
    # for k, v in y_pre2true.items():
    #     y_pre2true_dict[k] = {}
    #     vv = [i for i in v if i != []]
    #     for i in v:
    #         for j in i:
    #             y_pre2true_dict[k][j] = y_pre2true_dict[k].get(j, 0.0) + 1 / len(i) / len(vv)
    # y_pre2true_array, cid, tagList = dict2Array(y_pre2true_dict)
    # colorListAll = ['yellowgreen', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
    #                 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    #                 'chocolate',
    #                 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    #                 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    #                 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
    #                 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite',
    #                 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
    #                 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
    #                 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    #                 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
    #                 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
    #                 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
    #                 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
    #                 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
    #                 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    #                 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red',
    #                 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
    #                 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
    #                 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'aliceblue']
    # # colorList = random.sample(colorListAll, y_pre2true_array.shape[1])
    # colorList = [colorListAll[i * 4] for i in range(y_pre2true_array.shape[1])]
    # for i in range(y_pre2true_array.shape[1] - 1, -1, -1):
    #     plt.bar(cid, +y_pre2true_array[:, i], color=colorList[i], edgecolor='black', alpha=0.9)
    # plt.legend(tagList[::-1], bbox_to_anchor=(0, 1), loc=3, ncol=3, borderaxespad=0)
    # plt.show()
    # y_pre2true_array2 = y_pre2true_array.copy()
    # for i in range(1, y_pre2true_array2.shape[1]):
    #     y_pre2true_array2[:, i] += y_pre2true_array2[:, i - 1]
    # for i in range(y_pre2true_array.shape[1] - 1, -1, -1):
    #     plt.bar(cid, +y_pre2true_array2[:, i], color=colorList[i], edgecolor='black', alpha=1)
    # plt.legend(tagList[::-1], bbox_to_anchor=(0, 1), loc=3, ncol=3, borderaxespad=0)
    # plt.show()
    # # 评价聚类结果
    # # #评价指标
    # # 样本距离最近的聚类中心的距离总和
    # inertias = km.inertia_
    # print("%f\t样本距离最近的聚类中心的距离总和" % inertias)
    # # 轮廓系数：si值介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中，
    # # 越接近于0，表示样本应该在边界上。所有样本的si均值被称为聚类结果的轮廓系数。
    # silhouette_s = metrics.silhouette_score(data, y_pre, metric='euclidean')
    # print('%f\t轮廓系数：介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中,0表示样本应该在边界上' % silhouette_s)
    # silhouette_list = metrics.silhouette_samples(data, y_pre, metric='euclidean')
    # showDistplot(silhouette_list, title='样本轮廓系数分布')
    # # 获得外部评价
    # y_true = [i[0] for i in y_true]
    # # def analysis_restul(y_pre, y_true):
    # # 调整后的兰德指数，范围是[-1,1]，值越大意味着聚类结果与真实情况越吻合。
    # adjusted_rand_s = metrics.adjusted_rand_score(y_true, y_pre)
    # print('%f\t调整后的兰德指数，范围是[-1,1]，值越大意味着聚类结果与真实情况越吻合' % adjusted_rand_s)
    # # 互信息衡量聚类结果与实际类别的吻合程度，范围[0，1]，越接近1越吻合
    # mutual_info_s = metrics.mutual_info_score(y_true, y_pre)
    # print('%f\t互信息衡量聚类结果与实际类别的吻合程度，范围[0，1]，越接近1越吻合' % mutual_info_s)
    # # 调整后的互信息，范围[-1，1]，越接近1越吻合
    # adjusted_mutual_info_s = metrics.adjusted_mutual_info_score(y_true, y_pre)
    # print('%f\t调整后的互信息，范围[-1，1]，越接近1越吻合' % adjusted_mutual_info_s)
    # # 同质化得分（均一性）指每个簇中只包含单个类别的样本。
    # # 如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C|K)，值越大则均一性越小。
    # homogeneity_s = metrics.homogeneity_score(y_true, y_pre)
    # print('%f\t如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C|K)，值越大则均一性越小。' % homogeneity_s)
    # # 完整性（Completeness）指同类别样本被归类到相同的簇中。
    # # 如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K|C)，值越大则完整性越小。
    # completeness_s = metrics.completeness_score(y_true, y_pre)
    # print('%f\t如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K|C)，值越大则完整性越小。' % completeness_s)
    # # 单独考虑均一性或完整性都是片面的，因此引入两个指标的加权平均V - measure
    # v_measure_s = metrics.v_measure_score(y_true, y_pre)
    # print('%f\t均一性和完整性的的加权平均V - measure' % v_measure_s)
    #
    # # calinski&harabaz得分，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski - Harabasz分数会高
    # calinski_harabaz_s = metrics.calinski_harabaz_score(data, y_pre)
    # print('%f\t类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高' % calinski_harabaz_s)
    #
    # print('inertia\tARI\tMI\tAMI\thomo\tcomp\tv_m\tsilh\tc&h\n%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d' %
    #       (inertias, adjusted_rand_s, mutual_info_s, adjusted_mutual_info_s, homogeneity_s,
    #        completeness_s, v_measure_s, silhouette_s, calinski_harabaz_s))

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
