import shutil

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import os
from textClustringAnalysis.preprocessor.dealOCRTxt import doForOcr
from textClustringAnalysis.common import log
from textClustringAnalysis.preprocessor.dataInfo import dataInfo_main

'''
分词、去停顿词、词干提取
统计各词的词频

'''


def cleanData(txt: str) -> str:
    """清洗数据： 去除非法符号"""  # todo
    txt = doForOcr(txt)
    noabc = re.compile('[^a-zA-Z\']')
    txt = noabc.sub(' ', txt)
    letter = re.compile(' [a-zA-Z,.] ')
    txt = letter.sub(' ', txt)
    txt = txt.casefold()  # 小写
    return txt


def makeTag2pos(tag2pos):
    """建立pos_tag 标签 与 wnl.lemmatize的pos的映射 """
    if tag2pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wordnet.NOUN
    elif tag2pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wordnet.VERB
    elif tag2pos in ['JJ', 'JJR', 'JJS']:
        return wordnet.ADJ
    elif tag2pos in ['RB', 'RBR', 'RBS']:
        return wordnet.ADV
    else:
        return None


def preprocessing(txt, stop_words=None):  # todo
    """预处理： 分词 去停顿词 词干提取/词形还原"""
    txt = cleanData(txt)  # 清洗数据

    tokens = word_tokenize(txt)  # 分词 句子级
    tagged_sent = pos_tag(tokens)  # 获取单词词性 句子级
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = makeTag2pos(tag[1])  # 过滤掉 名词、动词、形容词、副词之外的词
        if wordnet_pos is None:
            continue
        newword = wnl.lemmatize(tag[0], pos=wordnet_pos)  # 按词性标签进行 词形还原

        # 停用词过滤
        if stop_words is not None and newword in stop_words:
            continue
        lemmas_sent.append(newword)

    r = ' '.join(lemmas_sent)
    return r


def dealOneTxt(filename, outfile, stop_words=None):
    """处理一个文件"""
    f = open(filename, 'r', errors='ignore')  # 读模式
    for line in f.readlines():
        line = line.replace('\x00\x00', ' ').replace('\x00', '')
        if re.match(r'-+Page [0-9]+-+', line) is not None:  # 页标行 -----------------------Page 7--------------
            continue
        newTxt = ''
        for sentence in line.split('. '):
            presentence = preprocessing(sentence, stop_words)
            if len(presentence) > 0:
                newTxt += preprocessing(presentence, stop_words) + '\n'
        with open(outfile, 'a', encoding='UTF-8') as f:
            f.write(newTxt)


def getStopWordSet(filename):
    with open(filename, 'r') as fr:
        stop_words = frozenset(fr.read().split())  # 将停用词读取到列表里
    return stop_words


@log('useTime')
def dealOneDir(inDir, outDir, stopWordFile=None):
    """处理一个文件夹"""
    files = os.listdir(inDir)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    stop_words = None
    if stopWordFile is not None:
        stop_words = getStopWordSet(stopWordFile)
    for file in files:
        if file.split('.')[-1] in ['txt', 'TXT']:
            inFile = inDir + '/' + file
            outFile = outDir + '/' + file
            if not os.path.exists(outFile):  # 已经存在则不再处理
                dealOneTxt(inFile, outFile, stop_words)


@log('Preprocessor_useTime')
def preprocessor_main(dirName, cache=False, q1=None, q2=None, outDir=None):
    """对文件集进行预处理 分词、去停顿词、词干提取"""

    cacheFiles = '%s_preproccess_cacheFiles' % dirName
    dealOneDir(dirName,
               cacheFiles,
               '/Users/brobear/PycharmProjects/TextClusteringAnalysis/textClustringAnalysis/preprocessor/stopwords.txt'
               )
    # 帅选
    outDir = dataInfo_main(cacheFiles, q1=q1, q2=q2, outDir=outDir)
    if not cache:  # 不保存缓存文件，即删除中间文件
        shutil.rmtree(cacheFiles)
    return outDir


if __name__ == '__main__':
    dirName = ''
    preprocessor_main(dirName)
