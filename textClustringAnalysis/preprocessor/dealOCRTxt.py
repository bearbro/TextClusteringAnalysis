
def doForOcr(txt: str) -> str:#TODO MORE
    """处理ocr识别产生的一些简单错误：0->O 8->B 6->b 1->I->i """
    m = {}
    m['0'] = 'O'
    m['8'] = 'B'
    m['6'] = 'b'
    m['1'] = 'I'
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    r = list(txt)
    lr = len(r)
    for i in range(lr):
        if r[i] in m:
            if (i > 0 and r[i - 1] in abc) or (i + 1 < lr and r[i - 1] in abc):
                r[i] = m[r[i]]
    r = ''.join(r)
    return r