import matplotlib.pyplot as plt
from numpy import mean
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

from textClustringAnalysis.feature.common import dict2Array
from textClustringAnalysis.feature.main import TC_PCA
from textClustringAnalysis.preprocessor.dataInfo import getWordCount, showDistplot
from textClustringAnalysis.preprocessor.start import preprocessor_main
from textClustringAnalysis.tag.getTag import getTags


def nbpj(data, y_pre):
    """内部评价"""
    # 样本距离最近的聚类中心的距离总和
    inertias = -1  # todo
    print("%f\t样本距离最近的聚类中心的距离总和" % inertias)
    # 轮廓系数：si值介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中，
    # 越接近于0，表示样本应该在边界上。所有样本的si均值被称为聚类结果的轮廓系数。
    silhouette_s = metrics.silhouette_score(data, y_pre, metric='euclidean')
    print('%f\t轮廓系数：介于[-1,1]，越接近于1表示样本i聚类越合理；越接近-1，表示样本i越应该被分类到其它簇中,0表示样本应该在边界上' % silhouette_s)
    silhouette_list = metrics.silhouette_samples(data, y_pre, metric='euclidean')
    showDistplot(silhouette_list, title='样本轮廓系数分布')
    # DB指数得分，分值越接近0越好
    davies_bouldin_s = davies_bouldin_score(data, y_pre)
    print('%f\tDB指数得分越接近0越好' % davies_bouldin_s)
    # calinski&harabaz得分，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski - Harabasz分数会高
    calinski_harabaz_s = metrics.calinski_harabaz_score(data, y_pre)
    print('%f\t类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高' % calinski_harabaz_s)


def wbpj(y_true, y_pre):
    """外部评价"""
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

    print('ARI\tMI\tAMI\thomo\tcomp\tv_m\n%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' %
          (adjusted_rand_s, mutual_info_s, adjusted_mutual_info_s, homogeneity_s,
           completeness_s, v_measure_s))


def show_wbpj(y_pre, textNames):
    """显示外部评价"""
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


def show_tsne(data, y_pre):
    # 在平面上显示聚类结果
    tsne = TSNE(n_components=2)
    a = tsne.fit_transform(data)  # 进行数据降维,降成两维
    import matplotlib.pyplot as plt
    wk = max(y_pre) + 1
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


def size_of_cluster(y_pre):
    """活动簇的元素个数"""
    kn = [0] * (max(y_pre) - min(y_pre) + 1)
    for i in y_pre:
        kn[min(y_pre) + i] += 1
    return kn


if __name__ == '__main__':
    ## 完整流程
    # # 预处理 about输入  1词性还原 2词干提取
    # outDir = preprocessor_main(dirName= '/Users/brobear/OneDrive/data-whitepaper/data/txt_about_all',  # 输入
    #                            cache='/Users/brobear/PycharmProjects/TextClusteringAnalysis/cache/txt_about_preproccess2',
    #                            # 中间结果保存
    #                            q1=50, q2=200,  # 筛选
    #                            outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt_about2')  # 结果输出
    # # pdf输入
    # outDir = preprocessor_main(dirName= '/Users/brobear/OneDrive/data-whitepaper/data/all_txt',  # 输入
    #                            cache='/Users/brobear/PycharmProjects/TextClusteringAnalysis/cache/all_txt_preproccess2',
    #                            # 中间结果保存
    #                            q1=1000, q2=8000,  # 筛选
    #                            outDir='/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2')  # 结果输出

    # 直接提供预处理后的文件
    # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt1'  # 词性还原
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'  # 词干提取
    # 获得词频
    txt_dict = getWordCount(outDir)
    # # tfidf矩阵
    # tfidf_dict = myTFIDF(txt_dict, itc=False)
    # tfidf_array, textNames, wordName = dict2Array(tfidf_dict)
    # 特征降维
    topN = 300
    data, textNames = TC_PCA(txt_dict, topN=topN)[:2]

    # k_Means聚类
    k = 10
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=9000)
    km.fit(data)
    y_pre = km.predict(data)
    # 层次聚类
    # todo
    # 密度聚类
    # todo

    # 在二维显示聚类结果
    show_tsne(data, y_pre)
    # 统计各个类的元素个数
    kn = size_of_cluster(y_pre)
    print(k, silhouette_score(data, y_pre, metric='euclidean'), max(kn), mean(kn), kn)

    # 评价聚类结果
    # 内部指标
    nbpj(data, y_pre)
    # 外部评价
    # 画图展示外部评价
    show_wbpj(y_pre=y_pre, textNames=textNames)
    # 获取标签
    # 构造外部指标 todo
    y_true_dict = getTags(textNames=textNames)
    y_true = ['None'] * len(textNames)  # 无标签则默认为None
    for i in range(len(y_true)):
        if len(y_true_dict[textNames[i]]) > 0:
            y_true[i] = y_true_dict[textNames[i]][0]
    # 输出外部指标
    wbpj(y_true=y_true, y_pre=y_pre)

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
