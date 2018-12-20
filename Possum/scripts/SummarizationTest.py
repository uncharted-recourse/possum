



# webpage corpus example
url = "http://newknowledge.com"
print("DEBUG::main::starting sumy url summarization test...")
TopicExtractor = Possum()
sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path=url,HTML=True,sentence_count=30)
print("These are the summary sentences:")
print(sentences)
extractedTopics = TopicExtractor.ExtractTopics(sentences)
print("These are the extracted topics, from the summary sentences: (sorted, importance weights included)")
print(extractedTopics)

# .csv file corpus example
# FIRST CLEAN TWEETS AND SUBSAMPLE FOR LOCAL MEMORY LIMITATION REASONS
# filename = "scripts/data/6_20_17_32_bp_content.txt"
filename = "scripts/data/pro_gun_text.csv"
print("DEBUG::main::starting sumy .csv file summarization test...")
TopicExtractor = Possum()
df = pd.read_csv(filename, dtype=str, header=None)
MAX_N_ROWS = 10000
df_list = df.ix[np.random.choice(df.shape[0],MAX_N_ROWS,replace=False),1].tolist() # subsample tweets
df_list = [TopicExtractor.clean_sentence(str(sentence)) + '.' for sentence in df_list] # clean tweets
# print("DEBUG::first 100 cleaned tweets:")
# print(df_list[:100])
(pd.DataFrame(df_list)).to_csv("scripts/data/truncated_frame.csv",index=False) # write cleaned tweets to file
# LOOK AT THE TOP 30 BIGRAMS
ngram_counts = Counter(ngrams("".join(df_cleaned_list).split(), 2))
print("The 30 most common bigrams are:")
print(ngram_counts.most_common(100))
# PERFORM SUMMARIZATION/TOPIC EXTRACTION
sentences = TopicExtractor.ExtractivelySummarizeCorpus(corpus_path="scripts/data/truncated_frame.csv",HTML=False,sentence_count=30)
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