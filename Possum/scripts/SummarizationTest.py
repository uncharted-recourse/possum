import numpy as np
import pandas as pd
from Possum import Possum
from collections import Counter


# A - webpage corpus example
url = "http://newknowledge.com"
print("DEBUG::main::starting sumy url summarization test...")
TopicExtractor = Possum()
sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path=url,HTML=True,sentence_count=30)
print("These are the summary sentences:")
print(sentences)
extractedTopics = TopicExtractor.ExtractTopics(sentences)
print("These are the extracted topics, from the summary sentences: (sorted, importance weights included)")
print(extractedTopics)

# B - .csv file corpus example
# 1 - FIRST CLEAN TWEETS AND SUBSAMPLE FOR LOCAL MEMORY LIMITATION REASONS
filename = "data/6_20_17_32_bp_content.txt"
# filename = "data/pro_gun_text.csv"
print("\n\n\nDEBUG::main::starting sumy .csv file summarization test...")
TopicExtractor = Possum(method='text_rank') # example non-default method specification
df = pd.read_csv(filename, dtype=str, header=None)
MAX_N_ROWS = 10000
df_list = df.ix[np.random.choice(df.shape[0],MAX_N_ROWS,replace=False),0].tolist() # subsample tweets
df_list = [TopicExtractor.clean_sentence(str(sentence)) + '.' for sentence in df_list] # clean tweets
# print("DEBUG::first 100 cleaned tweets:")
# print(df_list[:100])
(pd.DataFrame(df_list)).to_csv("data/truncated_frame.csv",index=False) # write cleaned tweets to file
# 2 - LOOK AT THE TOP 30 BIGRAMS
ngram_counts = Counter(TopicExtractor.ngrams(text="".join(df_list).split(), n=2))
print("The 30 most common word bigrams are:")
print(ngram_counts.most_common(30))
# 3 - PERFORM SUMMARIZATION/TOPIC EXTRACTION
sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path="data/truncated_frame.csv",HTML=False,sentence_count=30)
print("These are the summary sentences:")
print(sentences)
extractedTopics = TopicExtractor.ExtractTopics(sentences)
print("These are the topics extracted from the summary sentences: (sorted, importance weights included)")
print(extractedTopics)

'''
#df_full_list = df.tolist() # full list, with no subsampling
#df_full_list = [sentence + '.' for sentence in df_full_list]
sentences = df_list # extract topics from the whole corpus -> presently too expensive for large corpuses (sparsity etc.)
extractedTopics = TopicExtractor.ExtractTopics(sentences)
print("These are the extracted topics, from the whole corpus: (sorted, importance weights included)")
print(extractedTopics)'''