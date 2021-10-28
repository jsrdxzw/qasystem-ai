# -*- coding: UTF-8 -*-
from flask_restful import Resource, request
from flask import jsonify
from types import MethodType, FunctionType
import jieba
import re

from textrank4zh import TextRank4Keyword, TextRank4Sentence

from filter import DFAFilter
import fasttext.FastText as fasttext

tr4w = TextRank4Keyword(stop_words_file="stop_words.txt")
tr4s_sentence = TextRank4Sentence(stop_words_file="stop_words.txt")
gfw = DFAFilter()
gfw.parse("keywords")
classifier = fasttext.load_model("data_dim100_lr05_iter5.model")


class HttpCode(object):
    ok = 200
    un_auth_error = 401
    params_error = 400
    server_error = 500


def restful_result(code, message, data):
    return jsonify({"code": code, "message": message, "data": data})


def success(message="ok", data=None):
    return restful_result(HttpCode.ok, message=message, data=data)


# - content 要获取的内容
# - size    返回的个数
# - min_len 最小单词长度
class KeywordApi(Resource):
    def post(self):
        json_data = request.get_json()
        content = json_data.get('content')
        size = json_data.get('size', 10)
        min_len = json_data.get('min_len', 2)
        tr4w.analyze(text=content, lower=True, window=2)
        body = []
        for item in tr4w.get_keywords(size, word_min_len=min_len):
            res = {'word': item.word, 'score': item.weight}
            body.append(res)
        return success(data=body)


class PhraseApi(Resource):
    def post(self):
        """
            keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。
        """
        json_data = request.get_json()
        content = json_data.get('content')
        keywords_num = json_data.get('keywords_num', 20)
        min_occur_num = json_data.get('min_occur_num', 2)
        tr4w.analyze(text=content, lower=True, window=2)
        body = []
        for item in tr4w.get_keyphrases(keywords_num=keywords_num, min_occur_num=min_occur_num):
            res = {'phrase': item}
            body.append(res)
        return success(data=body)


class AbstractApi(Resource):
    def post(self):
        """获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
       Return:
       多个句子组成的列表。
        """
        json_data = request.get_json()
        content = json_data.get('content')
        sentence_num = json_data.get('num', 3)
        sentence_min_len = json_data.get('sentence_min_len', 6)
        tr4s_sentence.analyze(text=content, lower=True, source='all_filters')
        body = []
        for item in tr4s_sentence.get_key_sentences(num=sentence_num, sentence_min_len=sentence_min_len):
            res = {'index': item.index, 'sentence': item.sentence, 'score': item.weight}
            body.append(res)
        return success(data=body)


class FilterApi(Resource):
    def post(self):
        json_data = request.get_json()
        content = json_data.get('content')
        filter_content = gfw.filter(content, "*")
        return success(data=filter_content)


class AutoCategoryApi(Resource):
    __category_tag = {
        'Finance': '财经',
        'Lottery': '彩票',
        'Property': '房产',
        'Shares': '股票',
        'Furnishing': '家居',
        'Education': '教育',
        'Technology': '科技',
        'Sociology': '社会',
        'Fashion': '时尚',
        'Affairs': '时政',
        'Sports': '体育',
        'Constellation': '星座',
        'Game': '游戏',
        'Entertainment': '娱乐'
    }

    def __stop_words(self):
        with open('stop_words.txt', 'r', encoding='utf-8') as swf:
            return [line.strip() for line in swf]

    def __seg(self, sentence, sw, apply=None):
        if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
            sentence = apply(sentence)
        return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in sw])

    def __clean_txt(self, raw):
        fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
        return fil.sub(' ', raw)

    def post(self):
        json_data = request.get_json()
        content = json_data.get('content')
        labels = classifier.predict(
            [self.__seg(sentence=content.strip(), sw=self.__stop_words(), apply=self.__clean_txt)])
        label = str.split(labels[0][0][0], "__label__")[1]
        return success(data={"tag": label, "category": self.__category_tag[label]})
