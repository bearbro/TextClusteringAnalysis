from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pickle import load
'''
https://www.cnblogs.com/jclian91/p/9898511.html
'''


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


if __name__ == "__main__":
    sentence = 'football is    a family of team sports that involve, \n to varying degrees, kicking a ball to score a goal.'
    sentence='When proper inflation occurs due to additional issuance of SOMESING coins, this issuance can work as a tool to facilitate exchanges among major actors who own SOMESING coins, and to help prevent potential deflation causing destabilization of the economy.'

    tokens = word_tokenize(sentence)  # 分词

    tagged_sent = pos_tag(tokens)  # 获取单词词性
    print(tagged_sent)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1])  or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

    print(lemmas_sent)
    print(' '.join(lemmas_sent))
