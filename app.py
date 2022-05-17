from tracemalloc import start
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap

#NLP
from textblob import TextBlob,Word
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np
import re
import torch
import requests
from bs4 import BeautifulSoup
import time

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        rtext = request.form['rawtext']
        #NLP stuff
        blob = TextBlob(rtext)
        length = len(list(blob.words))
        #Sentiment Analysis
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        tokens = tokenizer.encode(rtext, return_tensors='pt')
        result = model(tokens)
        sentiment = int(torch.argmax(result.logits))
        sentiment_list = ["Negative sentance", "Below avg sentence", "Nice Sentence", "Above avg Sentence", "Positive Sentence"]
        sent = ""
        for i in range(5):
            if sentiment == i:
                sent = sentiment_list[i]   
        #Summarization
        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())      
                len_of_words = len(nouns)
                rand_words = random.sample(nouns,len(nouns))
                final_words = list()
                for item in rand_words:
                    word = Word(item).pluralize()
                    final_words.append(word)
                    summary = final_words
                    end = time.time()
                    final_time = end-start
    return render_template('index.html', rtext = rtext, length = length, sent = sent, summary=summary, final_time=final_time)

if __name__ == '__main__':
    app.run(debug=True)