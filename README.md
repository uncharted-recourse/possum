## Possum
Possum - Post Summarization 

Extractive corpus-level social media post summarization.

Input can be a .csv corpus derived from some social media platform or webpage url.

Extracts a given number of most important sentences (extracted from a co-occurence graph of the words in each sentence). Can additionally extract most important words from these top sentences using well-known tf-idf approaches. Also able to return word n-grams from the corpus if needed

Please be sure to clean your sentences are shown in the examples to ensure good results. Utilities for cleaning are provided

Sentences can be selected using a variety of algorithms (implementation from library `sumy` - https://github.com/miso-belica/sumy).

Specifically methods included are Luhn heuristic, Edmundson heurestic, Latent Semantic Analysis,
LexRank (default, unsupervised approach inspired by algorithms PageRank and HITS), TextRank (also a PageRank-type algorithm), SumBasic and KL-Sum (greedily add sentences to a summary so long as it decreases the KL Divergence). See the sumy link above for more information and sources.
Reduction 

## Installation

Perform the following two commands:

```bash
pip3 install git+https://github.com/NewKnowledge/possum
```
Alternatively, if the code is cloned locally (from `possum/` directory):

```bash
pip3 install .
```

## Examples
See the bottom of `Possum.py` for example code that serves as demonstration of capabilities. Indeed one can just do

```bash
python3.6 Possum.py
```

without pip installing to immediately test on one example (progun twitter text, using the default LexRank method), as well as a webpage. 

In the `/scripts` subfolder checkout sample code for library when used in pip installable form, in `SummarizationTest.py`, using a nondefault TextRank method, on another example (Black Panther Movie twitter data).

More detailed instructions forthcoming.
