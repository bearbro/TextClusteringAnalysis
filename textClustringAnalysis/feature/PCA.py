# 导入numpy库
import numpy
import numpy.matlib
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA

from textClustringAnalysis.feature.TC import doTC_dict
from textClustringAnalysis.feature.common import dict2Array
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.common import log


@log('useTime')
def myPCA_R(dataMat, changeR=False, m=0.85, topN=None, onlyNewData=True):
    """PCA"""
    # 求数据矩阵每一列的均值
    meanVals = numpy.mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列的特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵S
    S = numpy.cov(meanRemoved, rowvar=0)
    # 计算S的相关矩阵R
    if changeR:
        sqrtabsS = numpy.sqrt(numpy.abs(S))
        n = S.shape[0]
        R = numpy.matlib.empty(S.shape)
        for i in range(n):
            # todo 分母 <= 0 怎么办？？！！
            R[i, :] = S[i, :] / (sqrtabsS[i, :] * sqrtabsS[i, i])
    else:
        R = S
    # 计算R的特征值
    vals, vectors = numpy.linalg.eig(R)
    # 确定 topN
    valsIdx = numpy.argsort(vals)  # 返回特征值由小到大排序排序后的索引
    if topN is None:
        Fm = sum(vals) * m
        fmi = 0
        topN = 0
        while fmi < Fm:
            fmi += vals[valsIdx[-1 - topN]]
            topN += 1
    # 构造变换矩阵U
    eigValInd = valsIdx[:-1 - topN:-1]
    U = vectors[:, eigValInd]
    # 进行变换
    newData = dataMat * U
    if onlyNewData:
        return newData
    else:
        # 反构出原数据矩阵
        reconMat = (newData * U.T) + meanVals
        return newData, U, reconMat


@log("useTime")
def myPCA(dataMat, retenRate=0.85, topN=None, onlyNewData=True):  # (224,10000) 347s
    """PCA
    :type dataMat: 带处理矩阵
    :type retenRate: 保留率
    :type topN: 保留主成分个数
    :return: newData: 降维后矩阵, U:变换矩阵,reconMat:重构矩阵
    """
    # 标准化
    Smean = numpy.mean(dataMat, axis=0)
    Svar = numpy.var(dataMat, axis=0, ddof=1)
    Z = (dataMat - Smean) / Svar
    # 求相关矩阵
    n = dataMat.shape[0]
    R = (Z.T * Z) / (n - 1)
    # 求特征值
    vals, vectors = numpy.linalg.eig(R)  # 非常耗时间 99%
    # 确定个数
    valsIdx = numpy.argsort(vals)  # 返回特征值由小到大排序排序后的索引
    if topN is None:
        Fm = sum(vals) * retenRate
        fmi = 0
        topN = 0
        while fmi < Fm:
            fmi += vals[valsIdx[-1 - topN]]
            topN += 1
    # 构造变换矩阵U
    eigValInd = valsIdx[:-1 - topN:-1]
    U = vectors[:, eigValInd]
    # 进行变换
    newData = Z * U
    if onlyNewData:
        return newData
    else:
        # 反构出原数据矩阵
        reconMat = numpy.multiply((newData * U.T), Svar) + Smean
        return newData, U, reconMat


@log("useTime")
def pca_sklearn(dataMat, topN=None, onlyNewData=True):  # 必须保证 topNfeat>min(dataMat.shape)
    sklearn_pca = sklearnPCA(n_components=topN, svd_solver='full')
    # svd_solver {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
    sklearn_transf = sklearn_pca.fit_transform(dataMat)
    if onlyNewData:
        return sklearn_transf
    else:
        return sklearn_transf, None, sklearn_pca.inverse_transform(sklearn_transf)


def showDiff(data1, data2):
    pyplot.plot(data1[:, 0], data1[:, 1], 'o', markersize=7, color='blue', alpha=0.5,
                label='data1')
    pyplot.plot(data2[:, 0], data2[:, 1], '^', markersize=7, color='red', alpha=0.5,
                label='data2')
    pyplot.show()


def test_pca_sklearn():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')  # 0.6s
    txt_array, txtName, wordName = dict2Array(txt_dict, dtype=int)
    testdata = txt_array[:, 1:500]

    # newData_dict = doTC_dict(txt_dict, topN=1000)
    # testdata, txtName, wordName = dict2Array(newData_dict, dtype=int)
    dataMat = numpy.mat(testdata)

    print(dataMat.shape)
    topNfeat = 50
    lowDDataMat, redEigVects = myPCA_R(dataMat, topN=topNfeat)

    # showDiff(dataMat, lowDDataMat)

    # sklearn_transf, sklearn_pca = pac_sklearn(dataMat, topNfeat)
    # showDiff(lowDDataMat, sklearn_transf )

    # dataMat = pca.replaceNanWithMean()
    # meanVals = mean(dataMat, axis=0)
    # meanRemoved = dataMat - meanVals
    # covMat = cov(meanRemoved, rowvar=0)
    # eigVals, eigVects = linalg.eig(mat(covMat))


def getDiff(dataMat, rdata):
    dd = dataMat - rdata
    dd = numpy.multiply(dd, dd)
    return dd.sum()


def test_pca():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')  # 0.6s
    txt_array, txtName, wordName = dict2Array(txt_dict, dtype=int)
    dataMat = numpy.mat(txt_array[:, :1000], dtype=numpy.float64)
    topN = 100
    newData, U, rdata = myPCA(dataMat, topN=topN, onlyNewData=False)
    newData2, U2, rdata2 = myPCA_R(dataMat, topN=topN, onlyNewData=False)
    newData3, U3, rdata3 = pca_sklearn(dataMat, topN=topN, onlyNewData=False)
    # showDiff(newData, newData2)
    # dd = getDiff(dataMat, newData, U)
    # rdata = newData * U.T + numpy.mean(dataMat, axis=0)
    # print(dd,rdata.max())
    print(getDiff(dataMat, rdata))
    print(getDiff(dataMat, rdata2))
    print(getDiff(dataMat, rdata3))


if __name__ == '__main__':
    pass
