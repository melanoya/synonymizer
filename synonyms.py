# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
import gensim
import pymorphy2
import re
import csv
import os

model_1 = gensim.models.KeyedVectors.load_word2vec_format("./ruwikiruscorpora/model.bin", binary=True)
# model.init_sims(replace=True)
# model_news = gensim.models.KeyedVectors.load_word2vec_format("./news/model.bin", binary=True)
morph = pymorphy2.MorphAnalyzer()

tags = {'NOUN': 'NOUN', 'ADJF': 'ADJ', 'ADJS': 'ADJ', 'COMP': 'ADJ', 'VERB': 'VERB', 'INFN': 'VERB', 'PRTF': 'VERB', 'PRTS': 'VERB', 'GRND': 'VERB', 'NUMR': 'NUM', 'ADVB': 'ADV', 'NPRO': 'PRON', 'PRED': 'VERB', 'PREP': 'ADP', 'CONJ': 'CCONJ', 'PRCL': 'PART', 'INTJ': 'INTJ'}

def collectCapsulePath(PATH):
    arr = []
    capsules = os.listdir(PATH)
    train = r'\resources\ru-RU\training'
    for i in capsules:
        newPATH = PATH + '\\' + i + train
        if os.path.exists(newPATH) == True:
            arr.append([newPATH, i])
    return arr


def collectTrain(PATH):
#     capsule_name = re.findall('[a-zA-Z]+$', PATH)
#     PATH = PATH + r'\resources\ru-RU\training'
    files = os.listdir(PATH)
    arr = []
    for file_name in files:
        print(file_name)
        if file_name.endswith('.6t') or file_name.endswith('.bxb') == True:
            file = open(PATH + '\\' + file_name, 'r', encoding='utf-8').read()
            utterances = re.findall('utterance\(?"?(.*)"?', file)
#             utterances = re.findall('utterance \(?"?(.*)"?', file)
            for utterance in utterances:
                utterance = re.sub('\)$', '', utterance)
                utterance = re.sub('"$', '', utterance)
                arr.append(utterance)
    return arr


def getSynonyms(word, POS):
    search = word + '_' + POS
    arr = []
    for n in model_1.most_similar(positive=[search], topn=100):
        res = n[0].split('_')
        if res[1] == POS:
            count = model_1.wv.vocab[n[0]].count
            arr.append([res[0], count])
    return arr[:5]


def preprocessing(string, strPart):
    string = re.sub('\n', '', re.sub('^ ', '', re.sub('\[.*?\]', '', string)))
    if strPart == 'whole':
        whole_string = re.sub('[()]', '', string).split(' ')
        return whole_string
    elif strPart == 'in':
        inBrackets = ' '.join(re.findall('\((.*?)\)', string)).split(' ')
        return inBrackets
    elif strPart == 'outside':
        outBrackets = re.sub('  +', '', re.sub('\(.*?\)', '', string)).split(' ')
        return outBrackets


def changeSent(word, synonyms, line, POS, capsule_name):
    gram = morph.parse(word.lower())[0].tag
    gram_set = set(str(gram).replace(' ', ',').split(','))
    for synonym in synonyms:
#         print(synonym[1])
        try:
            new_word = morph.parse(synonym[0])[0].inflect(gram_set).word
            result = line.replace(word, new_word)
            outputRes(result, line, word, synonym[0], POS, capsule_name)
        except AttributeError:
            print(synonym[0], 'out of vocabulary')


def main(tags, POSs, PATH, capsule_name):
    file = collectTrain(PATH)
#     file = open(file_name, 'r', encoding='utf-8').readlines()
    for line in file:    
        string = preprocessing(line, 'whole')
        for word in string:
            POS = morph.parse(word.lower())[0].tag.POS
            if POS is not None and tags[POS] in POSs:
#                 print('POS pymorphy: ', POS)
#                 print('model ud POS: ', tags[POS])
                lemma = morph.parse(word.lower())[0].normal_form
                try:
                    synonyms = getSynonyms(lemma, tags[POS])
                    changeSent(word, synonyms, line, tags[POS], capsule_name)
                except KeyError:
                    print(word, 'out of vocabulary')


def outputRes(result, line, word, synonym, POS, capsule_name):
    line = re.sub('\n', '', line)
    f = open(capsule_name + '.csv', 'a', encoding='utf-8')
    f.write(POS + '\t' + word + '\t' + synonym + '\t' + line + '\t' + result + '\n')
#     print(result)


def main_2(PATHs, search):
    for PATH in PATHs:
        capsule_name = PATH[1]
        print(capsule_name)
        main(tags, search, PATH[0], capsule_name)


PATH = r'C:\Users\a.martynova.CORP\Documents\BIXBY\tools\bixbycapsules\primary'
search = ['ADJ']
main_2(collectCapsulePath(PATH), search)