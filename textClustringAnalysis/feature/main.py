import numpy

from textClustringAnalysis.feature.PCA import pca_sklearn
from textClustringAnalysis.feature.TC import doTC_dict
from textClustringAnalysis.feature.common import dict2Array, myTFIDF
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.common import log


@log("useTime")
def TC(txt_dict, topN):  # 7.6S
    newData_dict = doTC_dict(txt_dict, topN=topN)
    tfidf_dict = myTFIDF(newData_dict, itc=False)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    newData_mat = numpy.mat(tfidf_array)
    return newData_mat, txtName, wordName


@log("useTime")
def PCA(txt_dict, topN=None):  # 137s
    tfidf_dict = myTFIDF(txt_dict, itc=False)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    newData_mat = pca_sklearn(tfidf_array, topN=topN)
    return newData_mat, txtName


if __name__ == '__main__':
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    topN = 1000
    newData_mat, txtName, wordName = TC(txt_dict, topN)
    newData_mat2, txtName2 = PCA(txt_dict, topN=topN)
    numpy.savetxt('data_TC', newData_mat, delimiter=",")
    numpy.savetxt('data_PCA', newData_mat2, delimiter=",")
    # TEST
    txt_dict_test = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s_test' % 'afterProccess')
    newData_mat3, txtName3, wordName3 = TC(txt_dict_test, topN)
    numpy.savetxt('data_test', newData_mat3, delimiter=",")
