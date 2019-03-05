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

Sample input:
```I understand that you have been in contact with Jane Doe of our office.  She will be out of the office for the rest of the week and has asked that  I contact you regarding the proposed contract for differences between  our companies. She advises that she mentioned to you that Powerex  and Enron have recently completed two Contracts for Differences  in Alberta.  Enron has generated the confirms and attached Annex  A General Terms and Conditions.  We have since prepared a standard form  Contract for Differences which we would propose to use for future transactions. This document is more specifically designed for doing contract for differences  in Alberta.  Please review this document and provide us with your comments```


Sample output (`num_sentences=3`):
```['She will be out of the office for the rest of the week and has asked that  I contact you regarding the proposed contract for differences between  our companies.', 'Enron has generated the confirms and attached Annex  A General Terms and Conditions.', 'This document is more specifically designed for doing contract for differences  in Alberta.']```

## Possum Base Primitive
Possum - Post Summarization 
See https://github.com/NewKnowledge/possum.

Extractive corpus-level social media post summarization.

Input can be a .csv corpus derived from some social media platform or webpage url.

Extracts a given number of most important sentences (extracted from a co-occurence graph of the words in each sentence). Can additionally extract most important words from these top sentences using well-known tf-idf approaches. Also able to return word n-grams from the corpus if needed

Please be sure to clean your sentences are shown in the examples to ensure good results. Utilities for cleaning are provided

Sentences can be selected using a variety of algorithms (implementation from library `sumy` - https://github.com/miso-belica/sumy).

Specifically methods included are Luhn heuristic, Edmundson heurestic, Latent Semantic Analysis,
LexRank (default, unsupervised approach inspired by algorithms PageRank and HITS), TextRank (also a PageRank-type algorithm), SumBasic and KL-Sum (greedily add sentences to a summary so long as it decreases the KL Divergence). See the sumy link above for more information and sources.
Reduction 



