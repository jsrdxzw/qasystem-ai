import os
import re
from types import MethodType, FunctionType
import fasttext.FastText as fasttext

import jieba
from random import shuffle
import numpy as np

import pandas as pd


class _MD(object):
    mapper = {
        str: '',
        int: 0,
        list: list,
        dict: dict,
        set: set,
        bool: False,
        float: .0
    }

    def __init__(self, obj, default=None):
        self.dict = {}
        assert obj in self.mapper, \
            'got a error type'
        self.t = obj
        if default is None:
            return
        assert isinstance(default, obj), \
            f'default ({default}) must be {obj}'
        self.v = default

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, item):
        if item not in self.dict and hasattr(self, 'v'):
            self.dict[item] = self.v
            return self.v
        elif item not in self.dict:
            if callable(self.mapper[self.t]):
                self.dict[item] = self.mapper[self.t]()
            else:
                self.dict[item] = self.mapper[self.t]
            return self.dict[item]
        return self.dict[item]


mapper_tag = {
    '财经': 'Finance',
    '彩票': 'Lottery',
    '房产': 'Property',
    '股票': 'Shares',
    '家居': 'Furnishing',
    '教育': 'Education',
    '科技': 'Technology',
    '社会': 'Sociology',
    '时尚': 'Fashion',
    '时政': 'Affairs',
    '体育': 'Sports',
    '星座': 'Constellation',
    '游戏': 'Game',
    '娱乐': 'Entertainment'
}


def defaultdict(obj, default=None):
    return _MD(obj, default)


class TransformData(object):
    def to_csv(self, handler, output, index=False):
        dd = defaultdict(list)
        for line in handler:
            label, content = line.split(',', 1)
            dd[label.strip('__label__').strip()].append(content.strip())

        df = pd.DataFrame()
        for key in dd.dict:
            col = pd.Series(dd[key], name=key)
            df = pd.concat([df, col], axis=1)
        return df.to_csv(output, index=index, encoding='utf-8')


def split_train_test(source, auth_data=False):
    if not auth_data:
        train_proportion = 0.8
    else:
        train_proportion = 0.98

    basename = source.rsplit('.', 1)[0]
    train_file = basename + '_train.txt'
    test_file = basename + '_test.txt'

    handel = pd.read_csv(source, index_col=False, low_memory=False)
    train_data_set = []
    test_data_set = []
    for head in list(handel.head()):
        train_num = int(handel[head].dropna().__len__() * train_proportion)
        sub_list = [f'__label__{head} , {item.strip()}\n' for item in handel[head].dropna().tolist()]
        train_data_set.extend(sub_list[:train_num])
        test_data_set.extend(sub_list[train_num:])
    shuffle(train_data_set)
    shuffle(test_data_set)

    with open(train_file, 'w', encoding='utf-8') as trainf, \
            open(test_file, 'w', encoding='utf-8') as testf:
        for tds in train_data_set:
            trainf.write(tds)
        for i in test_data_set:
            testf.write(i)

    return train_file, test_file


def clean_txt(raw):
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)


def seg(sentence, sw, apply=None):
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in sw])


def stop_words():
    with open('stop_words.txt', 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]


def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch, lr=lr, wordNgrams=2,
                                               loss=loss)
    classifier.save_model(opt)
    return classifier


def cal_precision_and_recall(classifier, file='data_test.txt'):
    precision = defaultdict(int, 1)
    recall = defaultdict(int, 1)
    total = defaultdict(int, 1)
    with open(file) as f:
        for line in f:
            label, content = line.split(',', 1)
            total[label.strip().strip('__label__')] += 1
            labels2 = classifier.predict([seg(sentence=content.strip(), sw='', apply=clean_txt)])
            pre_label, sim = labels2[0][0][0], labels2[1][0][0]
            recall[pre_label.strip().strip('__label__')] += 1

            if label.strip() == pre_label.strip():
                precision[label.strip().strip('__label__')] += 1

    print('precision', precision.dict)
    print('recall', recall.dict)
    print('total', total.dict)
    for sub in precision.dict:
        pre = precision[sub] / total[sub]
        rec = precision[sub] / recall[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        print(f"{sub.strip('__label__')}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")


if __name__ == '__main__':
    # 对某个sentence进行处理：
    # content = '上海天然橡胶期价周三再创年内新高，主力合约突破21000元/吨重要关口。'
    content = ''
    with open('sports_test.txt') as f:
        content = f.read()
    res = seg(content.lower().replace('\n', ''), stop_words(), apply=clean_txt)
    print(res)
    # 转化成csv
    # td = TransformData()
    # __label__Shares , 中铁 物资
    # handler = open('data.txt')
    # td.to_csv(handler, 'data.csv')
    # handler.close()
    #
    # # 将csv文件切割，会生成两个文件（data_train.txt和data_test.txt）
    # train_file, test_file = split_train_test('data.csv', auth_data=True)

    # 训练
    # dim = 100
    # lr = 5
    # epoch = 5
    # model = f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'
    #
    # classifier = train_model(ipt='data_train.txt',
    #                          opt=model,
    #                          model=model,
    #                          dim=dim, epoch=epoch, lr=0.5
    #                          )
    #
    # result = classifier.test('data_test.txt')
    # print(result)

    # 预测
    classifier = fasttext.load_model("data_dim100_lr05_iter5.model")
    # cal_precision_and_recall(classifier)
    labels2 = classifier.predict([seg(sentence=content.strip(), sw='', apply=clean_txt)])
    print(labels2)