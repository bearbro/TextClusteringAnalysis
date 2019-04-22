from textClustringAnalysis.feature.TC import doTC_dict
from textClustringAnalysis.feature.common import dict2Array
from textClustringAnalysis.preprocessor.dataInfo import getWordCount
from textClustringAnalysis.common import log

# @log("useTime")
# def feature_main():
if __name__ == '__main__':
    txt_dict = getWordCount('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess_test')  # 0.6s
    newData_dict = doTC_dict(txt_dict, topN=1000)
    newData_array, txtName, wordName = dict2Array(newData_dict, dtype=int)
    print(newData_array.shape)
