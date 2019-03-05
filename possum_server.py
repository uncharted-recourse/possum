#
# GRPC Server for NK Possum text summarization primitive
# 
# Uses GRPC service config in protos/grapevine.proto
# 

import nltk.data
import nltk
from random import shuffle
from json import JSONEncoder
from flask import Flask, request

import time
import pandas
import pickle
import numpy as np
import configparser
import os.path
import os
import pandas as pd

import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures

import logging
from collections import Counter
from Possum import Possum

logger = logging.getLogger('nk_possum_server')
logger.setLevel(logging.DEBUG)

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_NUM_SENTENCES = 10
DEFAULT_NUM_SENTENCES = 3
LANGUAGE_ABBREVIATIONS = ['da','nl','en','fi','fr','de','hu','it','no','pt','ro','ru','es','sv']
LANGUAGES = ['danish','dutch','english','finnish','french','german','hungarian','italian','norwegian','portuguese','romanian','russian','spanish','swedish']
LANGUAGE_MAPPING = dict(zip(LANGUAGE_ABBREVIATIONS, LANGUAGES))
HTML_FLAG = False
ALGORITHMS = ['luhn','edmundson','lsa','text_rank','sum_basic','kl']

DEBUG = True # boolean to specify whether to print DEBUG information

#-----
class NKPossumSummarizer(grapevine_pb2_grpc.ExtractorServicer):

    def __init__(self):
        try:
            nltk.data.load('tokenizers/punkt/english.pickle')
        except Exception:
            logger.exception("Downloading NLTK tokenizers.")
            nltk.download('punkt')

    # Main extraction function
    def Extract(self, request, context):

        # init Extraction result object
        result = grapevine_pb2.Extraction(
            key = "summary_sentences",
            confidence=0.0,
            model="NK_possum_summarizer",
            version="0.0.1",
        )

        # Get text from input message.
        input_doc = request.text
        print("DEBUG:")
        print(input_doc)

        # Exception cases.
        if (len(input_doc.strip()) == 0) or (input_doc is None):
            return result

        # Check the language of the English. Use English 'en' as the default and fallback option.
        language_abbrev = request.language
        if language_abbrev not in LANGUAGE_ABBREVIATIONS:
            print("Unknown or unsupported language abbreviation. Using en = English.")
            language_abbrev = "en"

        if language_abbrev in LANGUAGE_MAPPING:
            language = LANGUAGE_MAPPING[language_abbrev]
        else:
            language = "english"
        

        # Parse number of summary sentences from input message "raw" field.
        try: 
            num_sentences = int(request.raw)
        except:
            print("Could not parse input.raw into an integer.")
            num_sentences = DEFAULT_NUM_SENTENCES

        # Cap the number of the sentences.
        num_sentences = min(num_sentences, MAX_NUM_SENTENCES)

        str_counter = Counter(input_doc)
        num_periods = str_counter["."]
        num_spaces = str_counter[" "]
        length_of_doc = len(input_doc)

        if num_periods < num_sentences or num_spaces < 2 or length_of_doc <= 3:
            print("Detected short input document. Setting num_sentences to 1.")
            num_sentences = 1

        start_time = time.time()
        
        if(DEBUG):
            print("DEBUG::input document:")
            print(input_doc)

        TopicExtractor = Possum(nltk_directory=NLTK_LOCATION)

        # Write the inputs to a temporary file to be processed.
        process_id = os.getpid()
        filename = 'temp_' + str(process_id) + '.txt'
        print(input_doc,  file=open(filename, 'w'))
        
        try:
            sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path=filename,HTML=HTML_FLAG,sentence_count=num_sentences)   
        except Exception:
            logger.exception("Problem extracting summary sentences.")
            raise Exception
        
        print(sentences)

        elapsed_time = time.time()-start_time
        print("Total time for summarization is : %.2f sec" % elapsed_time)
        
        # Include the summary sentences in the result object.
        try:
            result.values[:] = sentences 
        except Exception:
            logger.exception("Problem embedding summary sentences in result object.")
            raise Exception

        return result


#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ExtractorServicer_to_server(NKPossumSummarizer(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    nltk_directory = config['DEFAULT']['nltk_directory']
    print("nltk_directory " + nltk_directory)
    global NLTK_LOCATION
    NLTK_LOCATION = nltk_directory
    algorithm = config['DEFAULT']['algorithm']
    print("algorithm " + algorithm)
    global ALGORITHM
    ALGORITHM = algorithm
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    
    serve()