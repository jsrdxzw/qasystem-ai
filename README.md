### 文本智能分析系统

#### 使用方法
+ 自己训练模型
  1. 准备好语料，命名为data.txt，格式为__label__{类别} , {分词}
  2. 训练模型，详见`train_fasttext.py`
  3. 进入本项目下，执行docker镜像打包
     - `cd qasystem-ai`
     - `docker build --rm -t jsrdxzw/qaaisystem:v1 .`
     - `docker run -itd --name qaaisystem --restart always -p 5000:5000 jsrdxzw/qaaisystem:v1`
  4. 默认运行在5000端口
+ 也可以直接pull 已经打包好的镜像
  + `docker pull jsrdxzw/qaaisystem:v1`
  + `docker run -itd --name qaaisystem --restart always -p 5000:5000 jsrdxzw/qaaisystem:v1`

#### 文本关键词提取
本接口采用**textrank**算法，分词并计算有向图，计算词语的权重

POST /qasystem_ai/extract_keywords
```json
// request
{
   "content": "一段文本",
   "size": "返回的关键词个数", // 默认为10
   "min_len": "最小单词长度" // 默认为2
}
// response
{
   "code": "200", 
   "message": "ok", 
   "data": [
      {
         "score": "0.12", // 分数，越高则越重要
         "word": "人工智能" // 关键词
      }
   ]
}
```

#### 文本关键短语提取
POST /qasystem_ai/extract_phrase
```json
// request
{
   "content": "一段内容",
   "keywords_num": "10", // 表示可能的组合，默认最大为20
   "min_occur_num": "2" // 表示关键词出现的次数，默认为2
}
// response
{
   "code": "200", 
   "message": "ok", 
   "data": [
      {
         "pharse": "人工智能的应用" // 关键短语
      }
   ]
}
```

#### 文本关键语句提取，标题自动生成
POST /qasystem_ai/extract_abstract
```json
// request
{
   "content": "一段内容",
   "sentence_num": "10", // 返回的语句个数，默认3
   "sentence_min_len": "6" // 表示语句的最小长度，默认6个字
}
// response
{
   "code": "200", 
   "message": "ok", 
   "data": [
      {
         "index": "6", // 在愿文本中出现的位置
         "sentence": "生成的语句",
         "score": "0.12" // 得分
      }
   ]
}
``` 

#### 敏感词过滤
POST /qasystem_ai/filter
```json
// request
{
   "content": "一段内容" // 要过滤的文本你
}
// response
{
   "code": "200", 
   "message": "ok", 
   "data": "过滤后的内容" // **你*
}
``` 

#### 自动分类
采用`fastText`算法，并且加上了softmax层，对30w篇新闻进行训练，测试准确度约为95%
目前支持以下的自动分类：
```text
{
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
```
POST /qasystem_ai/category
```json
// request
{
   "content": "文章内容"
}
// response
{
   "code": "200", 
   "message": "ok", 
   "data": {
      "tag": "Entertainment",
      "category": "娱乐"
   }
}
``` 
当然也可以自己准备好语料，使用`train_fasttext.py`里面的训练代码进行训练