#!/usr/bin/env python
#
# Test GRPC client code for NK Possum text summarization primitive
#
#

from __future__ import print_function
import logging
import sys
import csv

import grpc
import configparser
import grapevine_pb2
import grapevine_pb2_grpc

logger = logging.getLogger('nk_possum_client')
logger.setLevel(logging.DEBUG)

DEBUG = True # boolean to specify whether to print DEBUG information

def run(input_file):

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ExtractorStub(channel)

    text_to_summarize = ""
    try:
        textf = open(input_file, 'r')
    except IOError:
        logger.exception("Cannot open input file.")
        sys.exit(0)
    for line in textf:
        text_to_summarize = text_to_summarize + " " + line

    print(text_to_summarize)
    testMessage = grapevine_pb2.Message(
        raw="3", # This field is interpreted as the number of summary sentences to return.
        language=LANGUAGE,
        text=text_to_summarize)

    try:
        extraction = stub.Extract(testMessage)
        summary_sentences = extraction.values
        if (DEBUG):
            logger.info("Summary sentences: ")
            logger.info(summary_sentences)
        output_file = 'summary_' + input_file
        with open(output_file, 'w') as o:
            fileWriter = csv.writer(o, delimiter='\t')
            for sentence in summary_sentences:
                fileWriter.writerow([sentence])
    except Exception:
        logger.exception("Problem running client to extract summary sentences.")
        raise Exception
    

if __name__ == '__main__':
    logging.basicConfig() 
    config = configparser.ConfigParser()
    config.read('config.ini')
    port_config = config['DEFAULT']['port_config']
    logger.info("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    input_file = sys.argv[1]
    if len(sys.argv) > 1:
        language = sys.argv[2]
    else:
        language = 'en'
    global LANGUAGE
    LANGUAGE = language
    run(input_file)