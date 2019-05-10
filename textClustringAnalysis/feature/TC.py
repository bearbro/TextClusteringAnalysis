import random
import numpy

from textClustringAnalysis.common import log
from textClustringAnalysis.feature.common import myTFIDF, dict2Array

from textClustringAnalysis.preprocessor.dataInfo import showDistplot, fiveNumber, getWordCount


@log("useTime")
def myTC_dict(tfidf, rtype=dict):  # 1.3s
    """根据tfidf求单词贡献度TC"""
    sumtfidf = {}  # 统计各特征的tfidf之和
    tc = {}
    for (txt, worddict) in tfidf.items():
        for (w, tfidfi) in worddict.items():
            sumtfidf[w] = sumtfidf.get(w, 0) + tfidfi

    for (txt, worddict) in tfidf.items():
        for (w, tfidfi) in worddict.items():
            tc[w] = tc.get(w, 0) + tfidfi * (sumtfidf[w] - tfidfi)
    if rtype == list:
        tcdict = list(tc.items())
        tcdict.sort()
        dd = [i[1] for i in tcdict]
        return dd
    return tc


@log("useTime")
def myTC_array(tfidf):  # 快 0.6s
    """根据tfidf矩阵求单词贡献度TC"""
    N, Q = tfidf.shape
    tc = numpy.zeros(Q)
    sumtfidf = sum(tfidf)  # 对列求和
    for i in range(N):
        tc += tfidf[i] * (sumtfidf - tfidf[i])
    return tc


@log("useTime")
def myTC(tfidf, rtype=dict):
    """返回TC向量"""
    if type(tfidf) == dict:
        return myTC_dict(tfidf, rtype=rtype)
    elif type(tfidf) == numpy.ndarray:
        return myTC_array(tfidf)
    else:
        raise NameError('输入的tfidf类型出错')


@log("useTime")
def selectData_dict(txtdict, wordName):  # 快
    """返回dict进行特征选择的结果，仅保留wordName中的特征"""
    wordNameSize = len(wordName)
    wordSet = set(wordName)
    newDict = {}
    for (txt, worddict) in txtdict.items():
        newDict[txt] = {}
        worddictSize = len(worddict)
        if worddictSize < wordNameSize:
            for (w, v) in worddict.items():
                if w in wordSet:
                    newDict[txt][w] = v
        else:
            for w in wordName:
                if w in worddict:
                    newDict[txt][w] = worddict[w]
    return newDict


@log("useTime")
def selectData_array(data, wordName, oldWordName, orderchange=True):
    """返回矩阵进行特征选择的结果，仅保留wordName中的特征列"""
    # 建立映射关系
    if orderchange:
        word2id = dict(zip(oldWordName, range(len(oldWordName))))
        wordid = [word2id[i] for i in wordName]
    else:
        wordSet = set(wordName)
        wordid = [i for i in range(len(oldWordName)) if oldWordName[i] in wordSet]
    # 筛选
    newData = data[:, wordid]

    return newData


@log("useTime")
def selectData(data, wordName, oldWordName=None, orderchange=True):
    """对data进行特征选择操作"""
    if type(data) == dict:
        return selectData_dict(data, wordName)
    elif type(data) == numpy.ndarray:
        if oldWordName is None:
            raise NameError('缺少参数oldWordName')
        return selectData_array(data, wordName, oldWordName=oldWordName, orderchange=orderchange)
    else:
        raise NameError('输入的data类型出错')


@log("useTime")
def selectFeature(tc, wordName, minTC=0, topN=None):
    """根据TC选择特征 输入的
    :type wordName: 按字典序升序排列
    :return: newWordName 按字典序升序排列
    """
    if topN is None:
        return [wordName[i] for i in range(len(tc)) if tc[i] > minTC]
    else:
        idx = tc.argsort()
        idx = idx[:-topN - 1:-1]
        newWordName = [wordName[i] for i in idx]
        newWordName.sort()
        return newWordName


@log("useTime")
def doTC_dict(txt_dict, minTC=0, topN=None):  # 快
    """进行TC特征降维"""
    # 计算各文本各词itc权值的tfidf矩阵
    tfidf_dict = myTFIDF(txt_dict, itc=True)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    # 计算各词的单词权值
    tc_array = myTC_array(tfidf_array)
    # 根据TC权值筛选单词
    newWordName = selectFeature(tc_array, wordName, minTC=minTC, topN=topN)
    # 根据新的单词集（特征集）压缩数据
    newData = selectData(txt_dict, newWordName)
    return newData


@log("useTime")
def doTC_array(txt_dict, minTC=0, topN=None):
    """进行TC特征降维"""
    # 计算各文本各词itc权值的tfidf矩阵
    txt_array, txtName, wordName = dict2Array(txt_dict, dtype=int)
    tfidf_dict = myTFIDF(txt_dict, itc=True)
    tfidf_array = dict2Array(tfidf_dict)[0]
    # 计算各词的单词权值
    tc_array = myTC_array(tfidf_array)
    # 根据TC权值筛选单词
    newWordName = selectFeature(tc_array, wordName, minTC=minTC, topN=topN)
    # 根据新的单词集（特征集）压缩数据
    newData = selectData(txt_array, newWordName, oldWordName=wordName, orderchange=False)
    return newData, newWordName


def test_selectFeature():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')
    tfidf_dict = myTFIDF(txt_dict, itc=True)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    tc_array = myTC_array(tfidf_array)
    minTC, topN = 0, 100
    tc = tc_array
    wordAndIdx = list(zip(wordName, tc))
    wordAndIdx.sort(key=lambda x: x[1], reverse=True)  # 按tc排序
    newWordName = [wordAndIdx[i][0] for i in range(topN)]
    newWordName.sort()

    idx = tc.argsort()
    idx = idx[:-topN - 1:-1]
    idx.sort()
    newWordName2 = [wordName[i] for i in idx]
    for j in range(len(newWordName2)):
        if newWordName2[j] != newWordName[j]:
            print("%d tcWordName[i]!=tcWordName_dict_array" % j)
            break


def test_selectData():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    txt_array, txtName, wordName = dict2Array(txt_dict)

    newWid = random.sample(range(0, len(wordName)), 300)
    newWid.sort()
    newWordname = [wordName[i] for i in newWid]
    sfdata = selectData_dict(txt_dict, newWordname)
    sfdata2 = selectData_array(txt_array, newWordname, oldWordName=wordName)
    sfdata22 = selectData_array(txt_array, newWordname, oldWordName=wordName, orderchange=False)
    # 当newWordname顺序改变时即orderchange=True
    # selectData_dict 比 selectFeature_array快
    sfdata1 = dict2Array(sfdata)[0]
    print(sum(sum(sfdata1 - sfdata2)))


def test_myTC():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    tfidf_dict = myTFIDF(txt_dict, itc=True)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    tc_dict = myTC_dict(tfidf_dict, rtype=list)
    tc_array = myTC_array(tfidf_array)
    # myTC_array 比 myTC_dict 快
    # print(sum(tc_dict - tc_array))
    # 可以借助dataInfo内的函数、查看TC的数据分布
    print(fiveNumber(tc_array))
    showDistplot(tc_array)


def test_doTC():
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')
    minTC, topN = 0, 10000
    tcData_array, tcWordName = doTC_array(txt_dict, minTC, topN)
    tcData_dict = doTC_dict(txt_dict, minTC, topN)
    tcData_dict_array, txtName, tcWordName_dict_array = dict2Array(tcData_dict, dtype=int)
    # doTC_dict 比 doTC_array 快
    for j in range(len(tcWordName)):
        if tcWordName[j] != tcWordName_dict_array[j]:
            print("%d tcWordName[i]!=tcWordName_dict_array" % j)
            break
    print(sum(sum(tcData_dict_array - tcData_array)))


if __name__ == '__main__':
    txt_dict = getWordCount('/Users/brobear/PycharmProjects/TextClusteringAnalysis/all_txt_preproccess')
    tfidf_dict = myTFIDF(txt_dict, itc=True)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    # 计算各词的单词权值
    tc_array = myTC_array(tfidf_array)
    showDistplot(tc_array)
    tc_array.sort()
    from matplotlib import pyplot as plt

    plt.plot(range(len(tc_array)), tc_array)
    plt.ylim(0, 200)
    plt.show()
