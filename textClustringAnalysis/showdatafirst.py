import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from textClustringAnalysis.feature.common import dict2Array, myTFIDF
from textClustringAnalysis.preprocessor.dataInfo import getWordCount

if __name__ == '__main__':
    """
    当我们想要对高维数据进行分类，又不清楚这个数据集有没有很好的可分性（即同类之间间隔小，异类之间间隔大）
    可以通过t - SNE投影到2维或者3维的空间中观察一下。如果在低维空间中具有可分性，则数据是可分的；
    如果在高维空间中不具有可分性，可能是数据不可分，也可能仅仅是因为不能投影到低维空间。
    """
    outDir = '/Users/brobear/PycharmProjects/TextClusteringAnalysis/txt2'
    txt_dict = getWordCount(outDir)
    tfidf_dict = myTFIDF(txt_dict, itc=False)
    data, textNames, wordName = dict2Array(tfidf_dict)
    tsne = TSNE(n_components=2)
    a = tsne.fit_transform(data)  # 进行数据降维,降成两维
    plt.scatter(a[:, 0], a[:, 1], s=2, alpha=1)
    plt.show()
