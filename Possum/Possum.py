# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:13:37 2017

@author: azunre
"""

# sumy imports
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer #default
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# 
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer

# other imports
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

from collections import Counter
from nltk.corpus import stopwords
import nltk
import sys
import logging

logger = logging.getLogger('nk_possum_server')
logger.setLevel(logging.DEBUG)

# Hyper settings...
LANGUAGE = "english" # english is the default language
DEBUG = False # don't print debug info by default

class Possum:
    def __init__(self,method=None, nltk_directory=None, language=None):
        if (language):
            logger.info("Setting language to " + language)
            LANGUAGE = language
        # Set the location of the nltk data directory for tokenizers, etc.
        if nltk_directory:
            nltk.data.path.append(nltk_directory)
            logger.info(nltk.data.path)
        try:
            self.stemmer = Stemmer(LANGUAGE)
        except Exception:
            logger.exception("Error loading nltk stemmer")
            raise Exception("Error loading nltk stemmer")
            
        self.summarizer = Summarizer(self.stemmer) # default
        if method:
            if(method=='luhn'):
                logger.info("Using the Luhn summarizer!")
                self.summarizer = LuhnSummarizer(self.stemmer)
            elif(method=='edmundson'):
                logger.info("Using the Edmundson summarizer!")
                self.summarizer = EdmundsonSummarizer(self.stemmer)  
            elif(method=='lsa'):
                logger.info("Using the LSA summarizer!")
                self.summarizer = LsaSummarizer(self.stemmer)
            elif(method=='text_rank'):
                logger.info("Using the Text Rank summarizer!")
                self.summarizer = TextRankSummarizer(self.stemmer)
            elif(method=='sum_basic'):
                logger.info("Using the Sum Basic summarizer!")
                self.summarizer = SumBasicSummarizer(self.stemmer)
            elif(method=='kl'):
                logger.info("Using the KL summarizer!")
                self.summarizer = KLSummarizer(self.stemmer)
            elif(method=='lex_rank'):
                logger.info("Using the LexRank summarizer!")
                self.summarizer = LexRankSummarizer(self.stemmer)
        #print(method)
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

    def ExtractivelySummarizeCorpus(self,corpus_path:str,HTML:bool = True,sentence_count:int = 20):

        if(HTML):
            self.parser = HtmlParser.from_url(corpus_path, Tokenizer(LANGUAGE))
        else:
            # or for plain text files
            self.parser = PlaintextParser.from_file(corpus_path, Tokenizer(LANGUAGE))
        
        sentences = self.summarizer(self.parser.document, sentence_count)
        
        if(DEBUG):
            # logger.info("DEBUG::ExtractivelySummarizeCorpus::these are all the parser.document.sentences")
            # logger.info(self.parser.document.sentences)
            logger.info("DEBUG::ExtractivelySummarizeCorpus::top n=%d sentences:"%sentence_count)
            for sentence in sentences:
                logger.info(str(sentence))
        sentences = [str(sentence) for sentence in sentences]
        
        return sentences

    def ExtractTopics(self,sentences:List[str])->dict:
        
        TfidfVec = TfidfVectorizer(stop_words='english')

        tfidf = TfidfVec.fit_transform(sentences)
        features = TfidfVec.get_feature_names()

        if(DEBUG):
            logger.info("DEBUG::ExtractTopics::all features (words)")
            logger.info(features)

        cos_similarity_matrix = (tfidf * tfidf.T).toarray()
        dense_tfidf = tfidf.todense()

        if(DEBUG):
            logger.info("DEBUG::ExtractTopics::tfidf matrix dimension:")
            logger.info(tfidf.shape)
            logger.info("DEBUG::ExtractTopics::dense tfidf matrix:")
            logger.info(dense_tfidf)
            logger.info("DEBUG::ExtractTopics::cos_similarity_matrix:")
            logger.info(cos_similarity_matrix)

        topics = []
        all_importance_weights = np.empty(0)
        for i in np.arange(tfidf.shape[0]):
            for j in np.arange(tfidf.shape[0]):
                if((i!=j) and (cos_similarity_matrix[i,j]>0)):
                    importance_weights_vec = [tfidf[i,k]*tfidf[j,k] for k in np.arange(tfidf.shape[1])]
                    topic_indices = np.array(importance_weights_vec)>0
                    importance_weights = np.array(importance_weights_vec)[topic_indices]
                    tmp = np.array(features)[topic_indices]
                    topics.extend(list(tmp))
                    all_importance_weights = np.append(all_importance_weights,importance_weights)
                    if(DEBUG):
                        logger.info(importance_weights)
                        logger.info("DEBUG::ExtractTopics::extracted topics:")
                        logger.info(tmp)
                        logger.info("DEBUG::ExtractTopics::extracted topic weights:")
                        logger.info(importance_weights)

        unique_topics = list(set(topics))   
        if(DEBUG):
            logger.info("DEBUG::ExtractTopics::concatenated extracted topics:")
            logger.info(topics)
            logger.info("DEBUG::ExtractTopics::concatenated extracted topic weights:")
            logger.info(all_importance_weights)
            logger.info("DEBUG::ExtractTopics::unique extracted topics:")
            logger.info(unique_topics)

        importance_weights = {}
        for topic in unique_topics:
            importance_weights[topic]=np.sum(all_importance_weights[np.array(topics)==topic])

        # normalize
        tmp = np.array(list(importance_weights.values()))
        # factor=1.0/sum(np.exp(tmp)) # useful if softmax is required
        importance_weights = {k: 1.0/(1.0+np.exp(-v)) for k, v in importance_weights.items() }
        # sort in decreasing order
        import operator
        sorted_importance_weights = sorted(importance_weights.items(), key=operator.itemgetter(1),reverse=True)
        
        return sorted_importance_weights

    def ngrams(self,text, n=2):
        # logger.info(text)
        # text = str(text)
        return zip(*[text[i:] for i in range(n)])

    def strip_non_ascii(self,string):
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)

    def clean_sentence(self,text):
        text = self.strip_non_ascii(text)

        stopWords = set(stopwords.words('english'))
        stopWords.add('&')
        stopWords.add('&amp')
        stopWords.add('rt')
        def is_banned(word):
            if word.startswith('http') or \
                word.startswith('@') or \
                word.startswith('#') or \
                word in stopWords:
                    return True
            return False

        # Replace named entities with First_Second word format
        text = text.replace('"','')
        text = text.replace("'","")
        text = ' '.join([''.join([c for c in x.lower()]) for x in text.split() if not is_banned(x)])

        text = str(text)

        return text

if __name__ == "__main__":
    nltk_location = sys.argv[1]
    # A - webpage corpus example
    url = "http://newknowledge.com"
    logger.info("DEBUG::main::starting sumy url summarization test...")
    if nltk_location:
        TopicExtractor = Possum(nltk_directory=nltk_location)
    else:
        TopicExtractor = Possum()
    sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path=url,HTML=True,sentence_count=30)
    logger.info("These are the summary sentences:")
    logger.info(sentences)
    extractedTopics = TopicExtractor.ExtractTopics(sentences)
    logger.info("These are the extracted topics, from the summary sentences: (sorted, importance weights included)")
    logger.info(extractedTopics)
    
    # B - text file corpus example
    filename = "data/NASA_TestData.txt"
    logger.info("\n\n\nDEBUG::main::starting sumy text file summarization test...")
    # example non-default method specification
    if nltk_location:
        TopicExtractor = Possum(method='lsa',nltk_directory=nltk_location)
    else:
        TopicExtractor = Possum(method='lsa')
    df = pd.read_csv(filename, dtype=str, header=None)
    #MAX_N_ROWS = 500
    #df_list = df.ix[np.random.choice(df.shape[0],MAX_N_ROWS,replace=False),1].tolist() # subsample tweets
    df_list = df.values.tolist()
    df_list = [TopicExtractor.clean_sentence(str(sentence)) + '.' for sentence in df_list] # clean tweets
    # 2 - LOOK AT THE TOP 30 BIGRAMS
    ngram_counts = Counter(TopicExtractor.ngrams(text="".join(df_list).split(), n=2))
    logger.info("The 30 most common word bigrams are:")
    logger.info(ngram_counts.most_common(30))
    # 3 - PERFORM SUMMARIZATION/TOPIC EXTRACTION
    sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path="data/NASA_TestData.txt",HTML=False,sentence_count=30)
    logger.info("These are the summary sentences:")
    logger.info(sentences)
    extractedTopics = TopicExtractor.ExtractTopics(sentences)
    logger.info("These are the topics extracted from the summary sentences: (sorted, importance weights included)")
    logger.info(extractedTopics)