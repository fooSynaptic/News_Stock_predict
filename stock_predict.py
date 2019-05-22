#py3

from jieba import cut
from multiclass_sl import Classifier

import random
from time import time
import numpy as np
import pandas as pd
import re
import random
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib


import keras
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Bidirectional, Input, LSTM, GlobalAveragePooling1D
from keras.models import Model, load_model, Sequential




import logging
logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        #filename='/home/URL/client/test_log.log',
                        filemode='a')

# Set parameters:
# ngram_range = 2 will add bi-grams features
global max_features
global model_path

ngram_range = 2
max_features = 80000
maxlen = 3000
batch_size = 128
embedding_dims = 1080
epochs = 100
model_path = './call_SVD.h5'


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


#preload chinese corpus and convert to char level
def preprocesing(texts):
    texts = [re.sub(r'[a-z0-9]|[—]', '', text.lower()).split() for text in texts]
    vocabs = set()
    for tokenset in texts:
        #text = re.sub(r'[ 0-9]', '', text)
        vocabs.update(tokenset)
    token_2_idx = {}
    idx_2_token = {}
    for idx, token in enumerate(vocabs):
        token_2_idx[token] = idx 
        idx_2_token[idx] = token
    return np.array([[token_2_idx[token] for token in text] for text in texts]), token_2_idx, idx_2_token







stopwords = [x.strip() for x in open('../data/stopwords/stop_words_zh.txt')\
      .readlines()]


def load_data(newspath, labelpath):
    #note two parameters are paths
    #first load labels
    #labels = np.array(pd.read_csv(labelpath).loc[:, ['trade_date','y']].iloc[:,:])
    #news = np.array(pd.read_csv(newspath).loc[:, ['date','content']].iloc[:,:])
    labels = np.array(pd.read_csv(labelpath).loc[:, ['trade_date', 'y']].iloc[:, :])
    news = np.array(pd.read_csv(newspath).loc[:, ['date', 'title']].iloc[:, :])

    texts, y = [], []

    for i in range(labels.shape[0]):
        if labels[i, 0] in news[:,0]:
            trade_time = labels[i][0]
            corpus = ''
            for j in range(news.shape[0]):
                if news[j, 0] == trade_time: corpus += str(news[j, 1])

            texts.append(corpus)
            y.append(float(labels[i, 1]))

    assert len(texts) == len(y), print('num of texts and labels do not alignmented...')


    #enumerate texts and labels and auxilary input
    auxi_y, main_text = [], []
    for i in range(len(texts)-2):
        main_text.append(''.join(texts[i:i+2]))
        auxi_y.append(y[i:i+2])

    y = y[-len(auxi_y):]

    assert len(main_text) == len(auxi_y) == len(y), print('num of texts and labels do not alignmented...')
    return tokenize_Input(main_text), y, auxi_y



def tokenize_Input(corpus):
    #iinput as list of dataset
    corpus = [x[:] for x in corpus]
    return [' '.join(cut(text)) for text in corpus]


def train(x, y = None):
    if not y:
        y = tagging(x)

    trainer = Classifier(x, y)
    return trainer

def model_eval(estimator, sample, labels):
    res = [int(estimator.infer(x)[0]) for x in sample]
    print(res[-10:])
    assert len(res) == len(labels), 'number not alignmented...'
    assert type(res[0]) == type(labels[0]), 'Type not alignmented{}vs{}'.format(type(res[0])\
        , type(labels[0]))
    result = [res[i]==labels[i] for i in range(len(res))]
    for predict, r, s, l in zip(res, result, sample, labels):
        if not r:
            pass

    return round(sum([res[i]==labels[i] for i in range(len(res))])/len(res), 3)


def data_augumental(news_file, stock_file):
    return load_data(news_file, stock_file)


#build keras model
def model(raw_data, fast = True):
    ngram_range = 2
    max_features = 80000
    
    batch_size = 128
    embedding_dims = 1080
    epochs = 500



    corpus, labels, auxiliary_input = raw_data
    print('Inspect raw data:',corpus[:5], labels[:5], auxiliary_input[:5])

    global token_2_idx
    global maxlen
    texts, token_2_idx, idx_2_token = preprocesing(corpus)
    maxlen = max([len(x) for x in texts])
 
    y = np.array([([1, 0] if x > 0.5 else [0, 1]) for x in labels])
    print('Inspecting labels:', y[:10])

    auxiliary_input = np.array(auxiliary_input)



    fold = int(texts.shape[0]*0.8)

    logging.info("We have total corpus number of {}, and fold at {} for test...".\
        format(texts.shape[0], fold))

    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    (x_train, auxi_train, y_train), (x_test, auxi_test, y_test) = (texts[:fold], auxiliary_input[:fold],\
     y[:fold]), (texts[fold:], auxiliary_input[fold:], y[fold:])


    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))


    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape, x_train[:3])
    print('x_test shape:', x_test.shape, x_test[:3])



    print('Build model...')

    if fast:
        #fast text style
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1, activation='sigmoid'))

        # categorical_crossentropy for multiclass labels
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logging.info('Start training...')

        if not os.path.exists(model_path):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test))
            logging.info('Save model...')
            model.save(model_path)
        

    else:
        #text processing && time-series input
        main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')

        vocab_size = len(token_2_idx.keys())
        x = Embedding(output_dim=embedding_dims, input_dim=vocab_size, input_length=maxlen)(main_input)

        lstm_out = Bidirectional(LSTM(100, dropout=0.3))(x)

        auxiliary_output = Dense(2, activation='softmax', name='aux_output')(lstm_out)

        auxiliary_input = Input(shape=(2,), name='aux_input')
        x = keras.layers.concatenate([lstm_out, auxiliary_input])

        main_output = Dense(2, activation='softmax', name='main_output')(x)

        model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

        #model.compile(optimizer='rmsprop', loss='binary_crossentropy',
        #          loss_weights=[0.4, 1], metrics=['accuracy'])

        model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy',
            loss_weights = [0.4, 1], metrics = ['accuracy'])


        logging.info('Start training...')

        if not os.path.exists(model_path) or True:
            model.fit([x_train, auxi_train], [y_train, y_train],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=([x_test, auxi_test], [y_test, y_test]))
            logging.info('Save model...')
            model.save(model_path)

        else:
            logging.info('Load from pretrained Model...')

            model = load_model(model_path)


def model_inspect(test_texts, labels, fold):
    print('Load model from path:{}'.format(model_path))
    model = load_model(model_path)

    #process test texts
    texts = np.array([[(token_2_idx[token] if token in token_2_idx else 999999) for token in text]\
     for text in test_texts])
    texts = sequence.pad_sequences(texts, maxlen=maxlen)

    click = False

    i = 0
    while not click:
        print("text", test_texts[fold:][i])
        res = model.predict(np.array([texts[fold:][i]]))
        print('infer result', res, ('牛市' if res[0][0] > res[0][1] else '熊市'))
        print("True label", ('牛市' if labels[fold:][i] == 1 else '熊市'))
        i += 1
        click = input()
        if click:
            print(model.predict(np.array(texts[fold:])))


def main():
    if not os.path.exists("./all_data.txt") or True:       
        x, y, auxi_y = load_data('./data/final_news.csv', './data/final_stock.csv')

        with open("./all_data.txt", 'w') as f:
            for a, b in zip(x, y):
                f.write(a+str(b)+'\n')
    else:
        corpus = [x.strip() for x in open('./all_data.txt').readlines()]
        x, y = [x[:-1] for x in corpus], [float(x[-1:]) for x in corpus]


    l = len(x)
    print('samples length:', l)
    model((x, y, auxi_y), fast = False)

    test_news, test_stock = pd.read_csv('./data/test_news.csv'), pd.read_csv('./data/test_stock.csv')
    corpus = [cut(x) for x in list(test_news['content'])]

    corpus = [[(token_2_idx[token] if token in token_2_idx else 999999) for token in text] for text in \
    corpus]
    texts = sequence.pad_sequences(corpus, maxlen=maxlen)

    finalmodel = load_model(model_path)
    #print(finalmodel.predict(texts))

main()
