#!/usr/bin/env python
# coding: utf-8

# In[151]:

import ast
import os 
import pandas as pd
import spacy
import re
import emot 
from string import punctuation
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import subprocess 
from ark_tweet_pos import CMUTweetTagger
import shlex
run_tagger_cmd = "java -XX:ParallelGCThreads=10 -Xmx500m -jar ark_tweet_pos/ark-tweet-nlp-0.3.2.jar"
ROOT_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.insert(0,'C:/Users/loren/Anaconda3/envs/tf_thesis/lib/site-packages/textblob')
sys.path.insert(1,'C:/Users/loren/Anaconda3/envs/tf_thesis/lib/site-packages')
sys.path.insert(2,'C:/Users/loren/Anaconda3/envs/tf_thesis/lib/site-packages/vaderSentiment')

from textblob import TextBlob
from sklearn.decomposition import PCA


# In[15]:


contractions = {
"ain't": "are not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"}


# In[16]:


def read_txt(path):
    global ROOT_DIR
    with open('{}\\{}'.format(ROOT_DIR,path),encoding='utf8') as f:
        data = f.readlines()
        f.close()
    return data

def read_csv(path,sep):
    global ROOT_DIR
    data = pd.read_csv('{}\\{}'.format(ROOT_DIR,path), sep = sep)
    return data


# In[123]:


class preprocessing_text:
    """
    txt_file: pandas df with the text column named 'text'
    language: type of language to use
    
    """
    def __init__(self, txt_file, language = 'en_core_web_sm',
                remove_mentions = True, remove_hashtags = True, lowercase = True, arktweet_pos = True):
        
        self.txt_file = txt_file
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.arktweet_pos = arktweet_pos
        self.nlp = spacy.load(language)
        self.stopwords = self.nlp.Defaults.stop_words
        self.initialism_list = []
        self.onomatopoeic_list = []
        
        for i in read_txt('Code/list_feature/initialism.list'):
            self.initialism_list.append(re.search('([A-Z]*[a-z]*)', i)[0])
    
        for i in read_txt('Code/list_feature/onomatopoeic.list'):
            self.onomatopoeic_list.append(re.search('([A-Z]*[a-z]*)', i)[0])
        
    def get_clean_df(self):
        
        df = self._preprocess_all(self.txt_file, self.remove_mentions, self.remove_hashtags, self.lowercase)
        return df
    
    def _preprocess_all(self, txt_file, remove_mentions = True, remove_hashtags = True, lowercase = True):

        if remove_mentions:
            txt_file['text'] = txt_file["text"].str.replace(r'@([^\s:]+)', '')

        if remove_hashtags:
            txt_file['text'] = txt_file['text'].str.replace(r'#([^\s:]+)', '')

        #Remove contractions and link (urls..)
        txt_file['text'] = txt_file['text'].apply(self.remove_link)
        txt_file['text'] = txt_file['text'].apply(self.remove_link2)
        txt_file['no_contr_text'] = txt_file.text.apply(self.remove_contraction)

        #Extract emoji infos
        txt_file['Emoji'] = txt_file.no_contr_text.apply(self.extract_emoji)

        #Remove useless spaces
        txt_file["no_contr_text"]  = txt_file["no_contr_text"].replace('\s+', ' ', regex=True)

        #Extract expressions
        txt_file['expressions_onomato'] = txt_file.no_contr_text.apply(self.extract_expressions, expression_list = self.onomatopoeic_list)
        txt_file['expressions_initialism'] = txt_file.no_contr_text.apply(self.extract_expressions, expression_list = self.initialism_list)

        #to lower case
        if lowercase:
            txt_file["no_contr_text"] = txt_file.no_contr_text.str.lower()

        #punctuaction
        txt_file['punctuation'] = txt_file.no_contr_text.apply(self.count_punctuation)

        #Remove all the stuff that aren't alphanumeric characters
        txt_file['removed_nowords'] = txt_file.no_contr_text.str.replace(r'(\W+)', ' ')

        #Get polarity and subjectivity information
        txt_file['polarity'] = txt_file.no_contr_text.apply(self.sentiment_info,type_sent = 0)
        txt_file['subjectivity'] = txt_file.no_contr_text.apply(self.sentiment_info,type_sent = 1)

        #Lemmatization, part of speech, name entity recognition
        lem, pos, ner = self.preprocess_pipe(txt_file['removed_nowords'], self.nlp, )

        txt_file['text_lemmatized'] = lem
        txt_file['text_lemmatized'] = txt_file['text_lemmatized'].apply(self.remove_comma)
        if self.arktweet_pos:
            txt_file['pos'] = CMUTweetTagger.runtagger_parse(txt_file.no_contr_text, run_tagger_cmd="java -XX:ParallelGCThreads=8 -Xmx500m -jar ../Code/ark_tweet_pos/ark-tweet-nlp-0.3.2.jar")
        else:
            txt_file['pos'] = pos
            
        txt_file['ner'] = ner

        return txt_file            
    
    @staticmethod
    def sentiment_info(text, type_sent = 0):
        '''
        type_sent: int 0/1, where 0 means get the polarity of the document, and 1 means get the subjectivity of the document
        '''
        text = text.encode('unicode-escape').decode('ASCII')

        if type_sent == 0:
            sentiment = TextBlob(text).sentiment[0]
        else:
            sentiment = TextBlob(text).sentiment[1]

        return sentiment
    
    @staticmethod
    def extract_expressions(x, expression_list):

        expression_diz = {k : 0 for k in expression_list}
        text = x.split(' ')

        for i in text:
            if i in expression_diz:
                expression_diz[i] += 1

        return expression_diz

    @staticmethod
    def extract_emoji(text):
        try:
            emoticons_list = emot.emoticons(text)['value']

        except TypeError:
            emoticons_list = []

        try:
            emoji_list = emot.emoji(text)['value']

        except TypeError:
            emoji_list = []
        emo_list = emoticons_list + emoji_list

        return emo_list
    
    @staticmethod
    def count_punctuation(text):

        counts = Counter(text)  # counts all occurences
        punct_diz = {k : 0 for k in punctuation}

        for i in counts:
            if i in punct_diz:
                punct_diz[i] = counts[i]

        return punct_diz
    
    @staticmethod
    def remove_comma(x):
        filtered = [i for i in x if i.strip()]
        return filtered    
    
    

    @staticmethod
    def preprocess_pipe(texts, nlp):
        
        def lemmatize_pipe(doc):
            lemma_list = [str(tok.lemma_) for tok in doc if not tok.is_stop] 
            return lemma_list

        def pos_pipe(doc):
            pos_list = [tok.pos_ for tok in doc]
            return pos_list

        def ner_pipe(x):
            ner_list = [token.label_ for token in x.ents]
            return ner_list        
        
        preproc_pipe_lemma = []
        preproc_pipe_pos = []
        preproc_pipe_ner = []

        for doc in nlp.pipe(texts, batch_size=20,  n_threads=12):
            preproc_pipe_lemma.append(lemmatize_pipe(doc))
            preproc_pipe_pos.append(pos_pipe(doc))
            preproc_pipe_ner.append(ner_pipe(doc))

        return preproc_pipe_lemma, preproc_pipe_pos, preproc_pipe_ner
           
    #clean url
    @staticmethod
    def remove_link(x):
        text = re.sub(r'^https?:\/\/.[\r\n]', '', x, flags=re.MULTILINE)
        return text

    #removes other link 
    @staticmethod
    def remove_link2(x):
        text = re.sub(r'http\S+', '', x)
        return text    
    @staticmethod
    def remove_contraction(text):
        for word in text.split():
            if word.lower() in contractions:
                text = text.replace(word, contractions[word.lower()])
        return text
    
    @staticmethod
    def remove_emoji(x):
        x = re.sub(r"(\<u+\S*>)", "", x)
        return x
    
    @staticmethod
    def map_nlp(x):
        x = nlp(x)
        return x


# In[169]:


class ExtractFeatures:
    """
    This class must be used after the preprocessing text phase 
    final_txt: pandas df extracted from the preprocessing class
    mod_pos: 'pos_sequences' (good if you want to take into account temporal dependecies), otherwise its a frequency matrix
    svd_transform: if all the features (except for polarity/subjectivity information) need to be transformed through svd
    """
    def __init__(self, final_txt, mod_pos, svd_transform = True):
        
        self.final_txt = final_txt
        self.mode_pos = mod_pos
        self.svd_all = svd_transform
        
    #extract all the features for training
    def get_all_features_train(self, ngram_range, dimensionality):
        
        punctuation_matrix = self.get_punctuaction()
        initialism_matrix = self.get_expres_initialism()
        onomato_matrix = self.get_expres_onomato()
        polarity_subj_matrix = self.polarity_subjectivity_features()
        emoji_matrix = self.CountVect('Emoji')
        self.emoji_list = [":'(", ":')", ':(', ':)',':*', ':-)', ':/', ':3', ':D', ':O', ':P', ';)', ';;', ';D', 'XD', 'XP', 'd:', '☕', '☹', '☺', '♥', '✋','✌',    '✔', '✨', '❄', '❤', '🎄', '🎶', '🏀', '👌', '👍','👎', '👏', '👑','👳', '💁', '💗', '😀', '😁', '😂', '😃','😄', '😅', '😆', '😇', '😉', '😊', '😋',  '😍', '😎', '😏', '😐', '😑', '😒', '😕', '😖', '😘', '😛', '😜', '😝', '😞', '😠', '😡', '😢', '😣', '😩', '😪', '😫', '😭', '😱', '😳', '😴', '😷', '😻', '🙈'] 
         #'😹','💙', ['😈', '😌', '😓'], ['✊', '🌝', '🌸', '🎈', '🎊', '🎤', '😌']
        
        self.positive_emoji = [":')", ':)', ':*',  ':-)',  ':3', ':D', ':O', 'XD', 'XP', 'd:','👍', '💁', '💗', '😀', '😁', '😂', '😃','😄', '😆', '😇','😊', '😋','😍', '😻'] #'😹','💙',
        
        self.negative_emoji = [":'(",':(',':/', '☹' ,'👎', '😐', '😑', '😒', '😔', '😕', '😖',  '😞', '😠', '😡', '😢', '😣', '😩', '😪', '😫', '😭', '😱','😷','🙈'] #['😈', '😌', '😓']
        
        self.onom_filter = ['haha', 'tweet', 'cough','la', 'click', 'ring', 'pop', 'bang', 'ugh','rush', 'hack', 'snap', 'slap','rip', 'flip', 'boo', 'knock', 'clip', 'bubble', 'bash', 'trickle', 'ping', 'clap', 'cock', 'mumble']
        self.init_filter = ['lol', 'LOL', 'OMG','LMAO', 'STFU', 'ROFL', 'lel', 'lolz','lul', 'GTFO', 'LMFAO', 'FML', 'SMH','FFS', 'ROFLMAO', 'lawl', 'FUBAR','lal', 'DGAF', 'SMFH', 'BL', 'BWL','OTF', 'BFD', 'ROTFLMFAO']
        
        self.punct_filtert = ['.',',','!', '"', '-', ':', '?','*', ')', "'", '^', '(','/', '_', ';', '&', '~','%', '=', '$']
        emoji_pos_matrix = emoji_matrix.reindex(columns = self.positive_emoji).sum(axis = 1)
        emoji_neg_matrix = emoji_matrix.reindex(columns = self.negative_emoji).sum(axis = 1)
        
        emoji_filter = emoji_matrix.reindex(columns = self.emoji_list)
        
        emoji_features = {'emoji_positive': emoji_pos_matrix, 'emoji_negative': emoji_neg_matrix, 'emoji': emoji_filter}
        
        pos_matrix = self.select_pos_representation() 
        #bow_matrix = self.bow() #prova
        if self.svd_all:
            tfidfsvd_word_matrix = self.tfidf_ngrams_svd_text(ngram_range, dimensionality)
            
            self.svdT2_punct = TruncatedSVD(n_components=10)
            svdTFit_punctuation = self.svdT2_punct.fit_transform(punctuation_matrix)
            
            self.emoji_svdT3 = TruncatedSVD(n_components=50)
            svdTFit_emoji = self.emoji_svdT3.fit_transform(emoji_matrix)
            
            self.svdT4_onomato = TruncatedSVD(n_components=10)
            svdTFit_onomato = self.svdT4_onomato.fit_transform(onomato_matrix)
            
            self.svdT5_initialism = TruncatedSVD(n_components=10)
            svdTFit_initialism = self.svdT5_initialism.fit_transform(initialism_matrix)
            
            return pos_matrix, tfidfsvd_word_matrix, svdTFit_punctuation, svdTFit_emoji, svdTFit_onomato, svdTFit_initialism, polarity_subj_matrix

        else:
            onomato_matrix = onomato_matrix.reindex(columns = self.onom_filter)
            initialism_matrix = initialism_matrix.reindex(columns = self.init_filter)
            punctuation_matrix = punctuation_matrix.reindex(columns = self.punct_filtert)
            return pos_matrix, punctuation_matrix,emoji_features,onomato_matrix,initialism_matrix, polarity_subj_matrix
    
    #use this after applied training features extraction
    def get_all_features_test(self, test_set):
        
        punctuation_matrix = pd.json_normalize(test_set["punctuation"])
        initialism_matrix = pd.json_normalize(test_set["expressions_initialism"])
        onomato_matrix = pd.json_normalize(test_set["expressions_onomato"])
        polarity_subj_matrix = test_set[['polarity','subjectivity']]
        #bow_matrix = self.bow(col = test_set['text_lemmatized'], test_fold = True)
        emoji_matrix = self.CountVect(col = test_set['Emoji'], test = True)
        
        emoji_pos_matrix = emoji_matrix.reindex(columns = self.positive_emoji).sum(axis = 1)
        emoji_neg_matrix = emoji_matrix.reindex(columns = self.negative_emoji).sum(axis = 1)
        
        emoji_filter = emoji_matrix.reindex(columns = self.emoji_list)
        
        emoji_features = {'emoji_positive': emoji_pos_matrix, 'emoji_negative': emoji_neg_matrix, 'emoji': emoji_filter}
        
        if self.mode_pos == 'pos_sequences':
            sequences_pos = self.pos_tokenizer.texts_to_sequences(test_set['pos'].astype(str))
            pos = pad_sequences(sequences_pos, maxlen=self.maxlen, padding='post')
        else: 
            try:
                pos = self.pos_vectorizer.transform(test_set.pos).todense()
            except AttributeError:
                new_pos = test_set.pos.apply(lambda x: ' '.join(x))
                pos = self.pos_vectorizer.transform(new_pos).todense()
        
        if self.svd_all:
            svdTFit_punctuation = self.svdT2_punct.transform(punctuation_matrix)
            svdTFit_emoji = self.emoji_svdT3.transform(emoji_matrix)
            svdTFit_onomato = self.svdT4_onomato.transform(onomato_matrix)
            svdTFit_initialism = self.svdT5_initialism.transform(initialism_matrix)
            try:
                tfs = self.word_tfidf.transform(test_set.text_lemmatized)
            except AttributeError:
                corpus = test_set.text_lemmatized.apply(lambda x: ' '.join(x))
                tfs = self.word_tfidf.transform(corpus)
            tfs = tfs.astype('float32')
            tfidfsvd_word_matrix = self.word_svdT.transform(tfs)            
            return pos, tfidfsvd_word_matrix, svdTFit_punctuation, svdTFit_emoji, svdTFit_onomato, svdTFit_initialism, polarity_subj_matrix
        else:
            onomato_matrix = onomato_matrix.reindex(columns = self.onom_filter)
            initialism_matrix = initialism_matrix.reindex(columns = self.init_filter)
            punctuation_matrix = punctuation_matrix.reindex(columns = self.punct_filtert)
            return pos, punctuation_matrix,emoji_features,onomato_matrix,initialism_matrix, polarity_subj_matrix
    
    #Choose the part of speech representation, pos_sequences is zero padded to the max lenght (sequence information)
    #otherwise is a count frequency matrix
#     def bow(self, col = 'text_lemmatized', test_fold = False):
#         if not test_fold:
#             corpus = self.final_txt.text_lemmatized.apply(lambda x: ' '.join(x))
#             self.cv_bow = CountVectorizer(dtype = np.int16)
#             bow_values = self.cv_bow.fit_transform(corpus).todense()
#         else:
#             corpus = col.apply(lambda x: ' '.join(x))
#             bow_values = self.cv_bow.transform(corpus).todense()

#         return bow_values
    
    def select_pos_representation(self):
        if self.mode_pos == 'pos_sequences':
            self.pos_tokenizer = Tokenizer(num_words=30)
            self.pos_tokenizer.fit_on_texts(self.final_txt['pos'].astype(str))
            sequences_pos = self.pos_tokenizer.texts_to_sequences(self.final_txt['pos'].astype(str))
            self.maxlen = 50
            pos = pad_sequences(sequences_pos, maxlen=self.maxlen, padding='post')
        else:
            self.pos_vectorizer = CountVectorizer(analyzer=lambda x: x)
            try:
                pos = self.pos_vectorizer.fit_transform(self.final_txt.pos).todense()
            except AttributeError:
                new_pos = self.final_txt.pos.apply(lambda x: ' '.join(x))
                pos = self.pos_vectorizer.fit_transform(new_pos).todense()
        return pos
    
    #extract tfidf from text and use svd for reducing the sparsity 
    def tfidf_ngrams_svd_text(self, ngram_range=(1,1), dimensionality = 2000):
        self.word_tfidf = TfidfVectorizer(stop_words = 'english', ngram_range=ngram_range)
        try:
            tfs = self.word_tfidf.fit_transform(self.final_txt.text_lemmatized)
        except AttributeError:
            corpus = self.final_txt.text_lemmatized.apply(lambda x: ' '.join(x))
            tfs = self.word_tfidf.fit_transform(corpus)
        tfs = tfs.astype('float32') 
        
        self.word_svdT = TruncatedSVD(n_components=dimensionality)
        svdTFit = self.word_svdT.fit_transform(tfs)
        return svdTFit
        
        
    def get_punctuaction(self):
        punctuation_matrix = pd.json_normalize(self.final_txt["punctuation"])
        return punctuation_matrix
    
    def get_expres_initialism(self):
        initialism_matrix = pd.json_normalize(self.final_txt['expressions_initialism'])
        return initialism_matrix
    
    def get_expres_onomato(self):
        onomato_matrix = pd.json_normalize(self.final_txt['expressions_onomato'])
        return onomato_matrix
    
    def polarity_subjectivity_features(self):
        polarity_subj_matrix = self.final_txt[['polarity','subjectivity']]
        return polarity_subj_matrix
    
    def CountVect(self, col, test=False):
        if not test:
            self.cv = CountVectorizer(analyzer=lambda x: x)
            counted_values = self.cv.fit_transform(self.final_txt[col]).toarray()
        else:
            counted_values = self.cv.transform(col).toarray()
            
        df = pd.DataFrame(counted_values, columns=self.cv.get_feature_names())
        return df 

