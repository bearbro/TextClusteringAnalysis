import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from textClustringAnalysis.feature.common import dict2Array, myTFIDF
from textClustringAnalysis.feature.main import TC, TC_PCA, PCA
from textClustringAnalysis.preprocessor.dataInfo import getWordCount

if __name__ == '__main__':
    """
    当我们想要对高维数据进行分类，又不清楚这个数据集有没有很好的可分性（即同类之间间隔小，异类之间间隔大）
    可以通过t - SNE投影到2维或者3维的空间中观察一下。如果在低维空间中具有可分性，则数据是可分的；
    如果在高维空间中不具有可分性，可能是数据不可分，也可能仅仅是因为不能投影到低维空间。
    """
    for i in ['txt1', 'txt2']:  # ['txt_about1','txt_about2']:
        # outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt1'
        outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/' + i
        txt_dict = getWordCount(outDir)
        # tfidf_dict = myTFIDF(txt_dict, itc=True)
        # data, textNames, wordName = dict2Array(tfidf_dict)
        data, textNames = TC_PCA(txt_dict, minTC=5, itc=False, topN=0.8)[:2]
        tsne = TSNE(n_components=2)
        a = tsne.fit_transform(data)  # 进行数据降维,降成两维
        plt.scatter(a[:, 0], a[:, 1], s=2, alpha=1)
        title = '%s TC5_PCA0_8' % i
        plt.title(title)
        plt.savefig('/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis/tsne-images/%s'
                    '.png' % title)
        plt.show()
# TC_PCA(txt_dict, minTC=5, itc=True,topN=0.8)[:2]  680
# TC_PCA(txt_dict, minTC=5, itc=False,topN=0.8)[:2] 497
# PCA(txt_dict, itc=False, topN=0.8)[:2] 1198
# PCA(txt_dict, itc=True, topN=0.8)[:2]  1171
# data, textNames = TC(txt_dict, topN=1100)[:2] 1100  txt1:37.64
# data, textNames = TC(txt_dict, topN=600)[:2] 600    txt1:47.00

# TC_PCA(txt_dict, minTC=5, itc=True,topN=0.8)[:2]  680

# minTC 0      5     10     37.64  38.67  47.00  52.74
# txt1  36731  3638  2684   1100           600
# txt2  29958  3305  2503          1100          600
