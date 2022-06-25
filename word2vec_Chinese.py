from gensim.models import Word2Vec
import jieba
import pandas as pd
import re
from stopwordsiso import stopwords


# input in csv format, each story added to list x as element
news = pd.read_csv('chinese_news.csv')
x = list(news['content'])

# input to word2Vec should be list containing lists of sentences
sentences = []
for col in x:
    # take the text out and remove white space
    text = str(col).rstrip()
    splitText = re.split('。|！|\!|\.|？|\?', text)
    for i in splitText:
        if i != '':
            sentences.append(i)

# pass through jieba to tokenize, then cut stopwords and punctuation
skipPunct = ['', '。', '，', '、', '：', '“', '”', "《", "》", '\n', '；', '— —', '（', '）']
normalizedSentences = []
for i in sentences:
    tokens = list(jieba.cut(i, cut_all=False))
    for t in tokens:
        if t in skipPunct or t in stopwords(['zh]']):
            tokens.remove(t)
    normalizedSentences.append(tokens)

# setup Word2Vec
model = Word2Vec(sentences=normalizedSentences, vector_size=100, window=3, min_count=3)
model.wv.similar_by_word('听', 10)