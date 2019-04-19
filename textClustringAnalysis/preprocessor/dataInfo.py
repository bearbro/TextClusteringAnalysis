import os
from start import log
import numpy
from matplotlib import pyplot
import shutil
import seaborn as sns
from scipy import stats

def dealOneTxt(filename):
    """处理一个文件"""
    wordset = set()
    txtlen = 0
    f = open(filename, 'r', encoding='utf-8', errors='ignore')  # 读模式
    for line in f.readlines():
        line = line.replace('\x00\x00', ' ').replace('\x00', '')
        for w in line.split():
            wordset.add(w)
            txtlen += 1
    return wordset, txtlen


@log("useTime")
def dealOneDir(inDir):
    """处理一个文件夹"""
    wordset = set()
    r = [[], [], []]
    files = os.listdir(inDir)
    for file in files:
        if file.split('.')[-1] in ['txt', 'TXT']:
            inFile = inDir + '/' + file
            wordseti, txtleni = dealOneTxt(inFile)
            wordset |= wordseti
            r[0].append(txtleni)
            r[1].append(len(wordseti))
            r[2].append(file)
    r.append(len(wordset))
    return r


@log("useTime")
def deleteFile(ADir, BDir):  # B-A
    Ailes = os.listdir(ADir)
    Biles = os.listdir(BDir)
    n = 0
    for A in Ailes:
        if A in Biles:
            os.remove(BDir + '/' + A)
            n += 1
    print(n)


def meanInfo(nums):
    # 均值
    a = numpy.mean(nums)
    # 中位数
    b = numpy.median(nums)
    counts = numpy.bincount(nums)
    # 返回众数
    c = numpy.argmax(counts)
    d = numpy.max(counts)
    f = numpy.std(nums)
    return ('均值：%f\n中位数:%f\n众数:%f %f\n标准差:%f' % (a, b, c, d, f))


def fiveNumber(nums):
    """五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）"""
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = numpy.percentile(nums, 25)
    Median = numpy.median(nums)
    Q3 = numpy.percentile(nums, 75)
    return Minimum, Q1, Median, Q3, Maximum


@log("useTime")
def selectFile(ADir, fileList, BDir):  # A->
    Ailes = os.listdir(ADir)
    if  os.path.exists(BDir):
        print("error: all")
        raise NameError('%s 已经存在' % BDir)
    os.mkdir(BDir)
    Biles = os.listdir(BDir)
    n = 0
    error = []
    for A in fileList:
        if A not in Ailes:
            error.append(A)
            continue
        if A not in Biles:
            shutil.copy(ADir + '/' + A, BDir + '/' + A)
            n += 1
    print(n)
    print("error:", error)


if __name__ == '__main__':
    # print(preprocessing('It’s 9% of all revenue after taxes generated through the HYGH platform from Day 1'))

    # dirName = 'txt_ocr_general_preproccess'
    info = dealOneDir('/Users/brobear/OneDrive/data-whitepaper/data/%s' % dirName)
    print('文本长度')
    print(meanInfo(info[0]))
    print(fiveNumber(info[0]))
    print('词特征')
    print(meanInfo(info[1]))
    print(fiveNumber(info[1]))
    print(info[3])
    #画数值的值分布图
    sns.distplot(info[0],kde=False, fit=stats.gamma)
    pyplot.show()

    print("帅选")
    # 去除长度不在【q1，q2】的数据
    q1 = 1000
    q2 = 8000
    info2id = [i for i in range(len(info[0])) if info[0][i] >= q1 and info[0][i] <= q2]

    info2 = [
        [info[0][k] for k in info2id],
        [info[1][k] for k in info2id]
    ]
    sns.distplot(info2[0],kde=False, fit=stats.gamma)
    pyplot.show()
    print('文本长度')
    print(meanInfo(info2[0]))
    print(fiveNumber(info2[0]))
    print('词特征')
    print(meanInfo(info2[1]))
    print(fiveNumber(info2[1]))
    print(info[3])
    print("筛选前:%d \n 筛选后:%d" % (len(info[0]), len(info2[0])))
    selectFile('/Users/brobear/OneDrive/data-whitepaper/data/%s' % dirName,
               [info[2][k] for k in info2id],
               '/Users/brobear/OneDrive/data-whitepaper/data/%s_90' % dirName
               )
