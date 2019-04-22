import numpy


from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.common import log


@log("useTime")
def dict2Array(txtdict, dtype=None):  # 2.5s
    """文本集向量化 vsm"""
    wordset = set()
    for (txt, worddict) in txtdict.items():
        wordset |= set(worddict.keys())

    txtName = list(txtdict.keys())
    wordName = list(wordset)
    txtName.sort()
    wordName.sort()
    wordNameMap = dict(zip(wordName, range(len(wordName))))
    if dtype is not None:
        data = numpy.zeros((len(txtName), len(wordName)), dtype=dtype)
    else:
        data = numpy.zeros((len(txtName), len(wordName)))
    for i in range(len(txtName)):
        for word, count in txtdict[txtName[i]].items():
            j = wordNameMap[word]
            data[i][j] = count

    return data, txtName, wordName


@log("useTime")
def myTFIDF_dict(txtdict, itc=False):  # 2.2s 1.6s
    """求各文本各词的tf-idf"""
    nk = {}
    for (txt, worddict) in txtdict.items():
        # 统计各特征的文本频率
        for w in worddict.keys():
            nk[w] = nk.get(w, 0) + 1

    N, Q = len(txtdict), len(nk)
    # 计算lg(N/nk)
    for (word, nkw) in nk.items():
        nk[word] = numpy.math.log10(N / nkw)

    tfidf = {}
    for (txt, worddict) in txtdict.items():
        tfidf[txt] = {}
        fm = 0

        for (word, tf) in worddict.items():
            if itc:  # lg(tf+1)* lg(N/nk)
                tfidf[txt][word] = numpy.math.log10(tf + 1) * nk[word]
            else:
                tfidf[txt][word] = tf * nk[word]  # tf * lg(N/nk)
            fm += tfidf[txt][word] ** 2
        fm = numpy.math.sqrt(fm)
        # 归一化
        for (word, tf) in worddict.items():
            tfidf[txt][word] /= fm

    return tfidf


@log("useTime")
def myTFIDF_array(data, itc=False):  # 129s 102s
    """利用特征矩阵求各文本各词的tf-idf"""
    N, Q = data.shape
    nk = numpy.zeros(Q)
    tfidf = numpy.asarray(data, dtype=float)
    # 统计各特征的文本频率
    for j in range(Q):
        sj = 0
        for i in range(N):
            if data[i][j] != 0:
                sj += 1
        nk[j] = sj
    # lg(N/nk)
    lgNnk = numpy.array(list(map(numpy.math.log10, (N / nk))))
    # 计算wik
    for i in range(N):
        #  lg(tf+1) * lg(N/nk)
        if itc:
            tfidf[i] = numpy.array(list(map(numpy.math.log10, tfidf[i] + 1))) * lgNnk
        else:
            tfidf[i] *= lgNnk  # tf * lg(N/nk)
        # 归一处理
        fm = numpy.math.sqrt(numpy.math.fsum(tfidf[i] ** 2))
        tfidf[i] /= fm

    return tfidf


@log("useTime")
def myTFIDF(data, itc=False):
    """返回tfidf矩阵"""
    if type(data) == dict:
        return myTFIDF_dict(data, itc=itc)
    elif type(data) == numpy.ndarray:
        return myTFIDF_array(data, itc=itc)
    else:
        raise NameError('输入的data类型出错')


@log("useTime")
def test_myTFIDF():
    txtdict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')  # 0.6s
    data, txtName, wordName = dict2Array(txtdict, dtype=int)
    tfidf2 = myTFIDF_array(data, itc=True)
    tfidf1 = myTFIDF(txtdict, itc=True)
    # myTFIDF_dict 比 myTFIDF_array 快
    dd = dict2Array(tfidf1)[0]
    cc = dd - tfidf2
    fdd = numpy.array([list(map(numpy.math.fabs, cci)) for cci in cc])
    print(sum(sum(fdd)))  # 误差在1e-15*n 浮点运算 精度下降


if __name__ == '__main__':
    pass

'''
sys.path.append('/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis/feature')
from start import *
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
txtdict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')
data, txtName, wordName = dict2Array(txtdict, dtype=int)
tfidf2=myTFIDF_array(data, itc=True)
tfidf1=myTFIDF(txtdict, itc=True)
dd=dict2Array(tfidf1)[0]
sum(sum(dd-tfidf2))
# numpy.savetxt('data', data, fmt="%d", delimiter=",")  # 占空间小，保存/加载速度慢 0.5G 45s 152s
# data2 = numpy.loadtxt('/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis/preprocessor/data.txt', delimiter=",")
# numpy.save('data', data)  # 占空间大，保存/加载速度快 1.67G 6s 2s
# data2 = numpy.load('data')
'''
