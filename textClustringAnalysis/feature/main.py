import numpy


from textClustringAnalysis.feature.PCA import pca_sklearn
from textClustringAnalysis.feature.TC import doTC_dict
from textClustringAnalysis.feature.common import dict2Array, myTFIDF
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.common import log


@log("Feature_useTime")
def TC(txt_dict, topN):  # 7.6S
    newData_dict = doTC_dict(txt_dict, topN=topN)
    tfidf_dict = myTFIDF(newData_dict, itc=False)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    newData_mat = numpy.mat(tfidf_array)
    return newData_mat, txtName, wordName


@log("Feature_useTime")
def PCA(txt_dict, topN=None, itc=False):  # 137s
    tfidf_dict = myTFIDF(txt_dict, itc=itc)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    newData_mat = pca_sklearn(tfidf_array, topN=topN)
    return newData_mat, txtName


@log("Feature_useTime")
def TC_PCA(txt_dict, minTC=0, topN=None, itc=False):  # 45s
    newData_dict = doTC_dict(txt_dict, minTC=minTC)
    tfidf_dict = myTFIDF(newData_dict, itc=itc)
    tfidf_array, txtName, wordName = dict2Array(tfidf_dict)
    newData_mat = pca_sklearn(tfidf_array, topN=topN)
    return newData_mat, txtName


if __name__ == '__main__':
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')  # 0.6s
    topN = 1800
    # newData_mat, txtName, wordName = TC(txt_dict, topN)
    # newData_mat2, txtName2 = PCA(txt_dict, topN=topN)
    newData_mat3, txtName3 = TC_PCA(txt_dict, minTC=0, topN=topN)
    # numpy.savetxt('data_TC_1800', newData_mat, delimiter=",")
    # numpy.savetxt('data_PCA_1800', newData_mat2, delimiter=",")
    numpy.savetxt('data_TC_PCA_1800', newData_mat3, delimiter=",")
    # TEST
    # txt_dict_test = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s_test' % 'afterProccess')
    # newData_mat3, txtName3, wordName3 = TC(txt_dict_test, topN)
    # numpy.savetxt('data_test', newData_mat3, delimiter=",")
