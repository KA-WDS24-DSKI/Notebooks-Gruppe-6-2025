import pandas as pd
import plotly.express as px
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --------------------------------------------------------------
# Download the stopwords from NLTK
nltk.download('stopwords')

df = pd.read_csv("Combined Data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

df.statement = df.statement.str.lower()
link_count = df.statement.str.count(r"https?://\S+")
link_count[link_count > 0]
df.statement = df.statement.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
df.statement = df.statement.str.replace(r"@\w+", "", regex=True)
df.statement = df.statement.str.replace(r"[^\w\s]", "", regex=True)
df.dropna(subset=["statement"], inplace=True)

#Tokenization
df["word_list"] = df.statement.str.split()
df.head()

#Word Instances too big -> maybe with nltk?
word_instance_head = np.concatenate(df["word_list"].head().values)
len(word_instance_head)

# --------------------------------------------------------------
word_counts = df.statement.str.split(expand=True).stack().value_counts()
word_counts = word_counts.reset_index()
word_counts.columns = ["word", "count"]
print("|V|=", len(word_counts.word))

word_counts.head()

# Barchart Top 20 Wörter
top_n = 20
top_words = word_counts.head(top_n)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(top_words["word"], top_words["count"])
plt.title(f"Top {top_n} häufigste Wörter")
plt.xlabel("Wort")
plt.ylabel("Anzahl")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
stop_words = set(stopwords.words("english"))
print(stop_words)

pattern = r"\b(" + "|".join(stop_words) + r")\b" # regex für stop words
df.statement = df.statement.str.replace(pattern, "", regex=True) # Stopwords entfernen

df.head()

df["word_list"] = df.statement.str.split()
df.head()

# Worte abzählen
word_counts = df.statement.str.split(expand=True).stack().value_counts()
word_counts = word_counts.reset_index()
word_counts.columns = ["word", "count"]
print("|V|=", len(word_counts.word))

# Häufigste Wörter:
word_counts.head()

# Barchart Top 20 Wörter
top_n = 20
top_words = word_counts.head(top_n)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(top_words["word"], top_words["count"])
plt.title(f"Top {top_n} häufigste Wörter")
plt.xlabel("Wort")
plt.ylabel("Anzahl")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace() # zunächst Trennung durch Leerzeichen
trainer = BpeTrainer(vocab_size=30000) # k=3000

# Liste mit Texten
tokenizer.train_from_iterator(df.statement, trainer) # tokenizer.train_from_iterator(iterator, trainer) - iteriere über die Tweets und verwende BPE als Trainer

# Text tokenisieren und Token zählen
all_tokens = []
for sentence in df.statement: # iteriere durch jede Zeile der Tweets, sentence1 = erste Zeile in df.Tweet
    encoded = tokenizer.encode(sentence) # Elemente des sentence werden Tokenisiert
    all_tokens.extend(encoded.tokens) # .extend() fügt Elemente an das Ende einer Liste an

print("Beispiel encoded: ", encoded.tokens)

# Häufigkeiten 
token_series = pd.Series(all_tokens) # umwandeln der Liste als Pandas Series
token_freq = token_series.value_counts()  # zählt Token
top_n = 15
top_tokens = token_freq.head(top_n)

# Barplot mit matplotlib
plt.figure(figsize=(10, 6))
plt.bar(top_tokens.index, top_tokens.values)
plt.title(f"Top {top_n} häufigste BPE-Token")
plt.xlabel("Token")
plt.ylabel("Häufigkeit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()