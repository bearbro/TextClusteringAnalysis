import os
import pymysql


def getTags(textNames):
    page = []
    subN = []
    tagmap = {}
    for name in textNames:
        pagei, subNi = list(name.split('-', maxsplit=2))[:2]
        page.append(pagei)
        subN.append(subNi)
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "admin", "icobenchdb")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    for i in range(len(page)):
        cursor.execute("SELECT tags from icos_tag where page =%s and subN=%s" % (page[i], subN[i]))
        # 使用 fetchone() 方法获取单条数据.
        data = cursor.fetchone()
        tagi = list(data[0].split(','))
        tagmap[textNames[i]] = tagi
    return tagmap


def getOneTag(textNames):
    """取第一个标签"""
    tagmap = getTags(textNames)
    for (v, k) in tagmap.items():
        tagmap[v] = k[0]
    return tagmap


def tag2id(textNames, tagmap):
    """将标签转换成数字符"""
    tagSet = set()
    if type(tagmap[textNames[0]]) is list:
        for i in tagmap.values():
            for j in i:
                tagSet.add(j)
    else:
        for i in tagmap.values():
            tagSet.add(i)
    setmap = {}
    id = 0
    tagid = []
    for i in tagSet:
        setmap[i] = id
        id += 1
    if type(tagmap[textNames[0]]) is list:
        for i in textNames:
            tagidi = [setmap[j] for j in tagmap[i]]
            tagid.append(tagidi)
    else:
        for i in textNames:
            tagid.append(setmap[tagmap[i]])
    return tagid


if __name__ == '__main__':
    dir = ('/Users/brobear/OneDrive/data-whitepaper/data/%s' % 'afterProccess')
    files = os.listdir(dir)
    page, subN, name = [], [], []
    tag = []
    for file in files:
        file = file[:-4]
        pagei, subNi, namei = list(file.split('-', maxsplit=2))
        page.append(pagei)
        subN.append(subNi)
        name.append(namei)
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "admin", "icobenchdb")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    for i in range(len(page)):
        cursor.execute("SELECT tags,name from icos_tag where page =%s and subN=%s" % (page[i], subN[i]))
        # 使用 fetchone() 方法获取单条数据.
        data = cursor.fetchone()
        if data[1] != name[i]:
            print(data[1], name[i])
        tagi = list(data[0].split(','))
        tag.append(tagi)
    # # 获取所有记录列表
    # results = cursor.fetchall()
    # 关闭数据库连接
    db.close()
    # tag种类
    tagset = {}
    tagsetM = {}
    for i in tag:
        # 所有标签
        for j in i:
            tagset[j] = tagset.get(j, 0) + 1  # 29
        # 主标签
        tagsetM[i[0]] = tagsetM.get(i[0], 0) + 1  # 29
# tagsetM==tagset:
# ['Art', 'Artificial Intelligence', 'Banking', 'Big Data', 'Business services', 'Casino & Gambling',
#  'Charity', 'Communication', 'Cryptocurrency', 'Education', 'Electronics', 'Energy', 'Entertainment',
#  'Health', 'Infrastructure', 'Internet', 'Investment', 'Legal', 'Manufacturing', 'Media', 'Other',
#  'Platform', 'Real estate', 'Retail', 'Smart Contract', 'Software', 'Sports', 'Tourism', 'Virtual Reality']
