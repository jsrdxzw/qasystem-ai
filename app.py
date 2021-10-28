#!/usr/bin/python
# -*- coding: UTF-8 -*-

from flask import Flask
from flask_restful import Api

from api import KeywordApi, PhraseApi, AbstractApi, FilterApi, AutoCategoryApi

app = Flask(__name__)
api = Api(app)
api.add_resource(KeywordApi, "/qasystem_ai/extract_keywords")
api.add_resource(PhraseApi, "/qasystem_ai/extract_phrase")
api.add_resource(AbstractApi, "/qasystem_ai/extract_abstract")
api.add_resource(FilterApi, "/qasystem_ai/filter")
api.add_resource(AutoCategoryApi, "/qasystem_ai/category")


@app.route("/", methods=['GET'])
def hello_world():
    return "qasystem ai!"
