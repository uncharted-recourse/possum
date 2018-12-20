from collections import Counter
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def ngrams(text, n=2):
        return zip(*[text[i:] for i in range(n)])

def strip_non_ascii(string):
   stripped = (c for c in string if 0 < ord(c) < 127)
   return ''.join(stripped)

def clean_sentence(text):
    text = strip_non_ascii(text)

    # stopwords = ['&','&amp;']
    # stopWords += [l.strip() for l in open('stops.txt').readlines()]
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
    filename = "data/pro_gun_text.csv"
    print("DEBUG::starting tweet clean and ngram extract test...")
    df = pd.read_csv(filename, dtype=str, header=None)
    df_cleaned_list = df.values[:,1].tolist()
    df_cleaned_list = [clean_sentence(str(sentence)) + '.' for sentence in df_cleaned_list]
    print("DEBUG::corpus before cleaning (pandas frame):")
    print(df)
    print("DEBUG::corpus after cleaning (list of first 100 tweets):")
    print(df_cleaned_list[:100])
    ngram_counts = Counter(ngrams("".join(df_cleaned_list).split(), 2))
    print("The 100s most common ngrams are:")
    print(ngram_counts.most_common(100))