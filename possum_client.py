#
# Test GRPC client code for NK Possum text summarization primitive
#
#

from __future__ import print_function
import logging

import grpc
import configparser
import grapevine_pb2
import grapevine_pb2_grpc

logger = logging.getLogger('nk_possum_client')
logger.setLevel(logging.DEBUG)

DEBUG = True # boolean to specify whether to print DEBUG information

def run():

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ExtractorStub(channel)

    testMessage = grapevine_pb2.Message(
        raw="3", # This field is interpreted as the number of summary sentences to return.
        language="en",
        text="I understand that you have been in contact with XXX XXX of our office. \
 She will be out of the office for the rest of the week and has asked that \
 I contact you regarding the proposed contract for differences between \
 our companies. She advises that she mentioned to you that Powerex \
 and Enron have recently completed two Contracts for Differences \
 in Alberta.  Enron has generated the confirms and attached Annex \
 A General Terms and Conditions.  We have since prepared a standard form \
 Contract for Differences which we would propose to use for future transactions.\
 This document is more specifically designed for doing contract for differences \
 in Alberta.  Please review this document and provide us with your comments")

    try:
        extraction = stub.Extract(testMessage)
        summary_sentences = extraction.values
        if (DEBUG):
            logger.info("Summary sentences: ")
            logger.info(summary_sentences)
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
    run()