## Possum Text Summarization GRPC Wrapper
Extractive corpus-level social media post summarization.

This repository provides a GRPC interface to New Knowledge's Possum text summarization primitive.

## Installation

Perform the following two commands:

```bash
pip3 install git+https://github.com/uncharted-recourse/possum
```

## Example of Running the Code

Make any needed changes to the `config.ini` file, for example, to change the summarization algorithm.

Start the server:
```python3.6 possum_server.py```

In a separate terminal session, run the client example:
```python3.6 possum_client.py```

* Input messages are instances of the `Message` class.
* Summarization results are included in the `result` as an instance of the `Extraction` class. See https://github.com/uncharted-recourse/grapevine/blob/master/grapevine/grapevine.proto. 


# gRPC Dockerized Summarization Server

The gRPC interface consists of the following components:
*) `grapevine.proto` in `protos/` which generates `grapevine_pb2.py` and `grapevine_pb2_grpc.py` according to instructions in `protos/README.md` -- these have to be generated every time `grapevine.proto` is changed
*) `spam_clf_server.py` which is the main gRPC server, serving on port `50052` (configurable via `config.ini`)
*) `possum_client.py` which is an example script demonstrating how the main gRPC server can be accessed to classify emails 
 
To build corresponding docker image:
`sudo docker build -t docker.ased.uncharted.software/nk-possum-text-summarization-binary:latest .`

To run docker image, simply do
`sudo docker run -it -p 50052:50052 docker.ased.uncharted.software/nk-email-classifier:latest`

Finally, edit `possum_client.py` with example email of interest for classification, and then run that script as
`python3 possum_client.py`

## Possum Base Primitive
Possum - Post Summarization 
See https://github.com/NewKnowledge/possum.

Extractive corpus-level social media post summarization.

Input can be a .csv corpus derived from some social media platform or webpage url.

Extracts a given number of most important sentences (extracted from a co-occurence graph of the words in each sentence). Can additionally extract most important words from these top sentences using well-known tf-idf approaches. Also able to return word n-grams from the corpus if needed

Sentences can be selected using a variety of algorithms (implementation from library `sumy` - https://github.com/miso-belica/sumy).

Specifically methods included are Luhn heuristic, Edmundson heurestic, Latent Semantic Analysis,
LexRank (default, unsupervised approach inspired by algorithms PageRank and HITS), TextRank (also a PageRank-type algorithm), SumBasic and KL-Sum (greedily add sentences to a summary so long as it decreases the KL Divergence). See the sumy link above for more information and sources.




