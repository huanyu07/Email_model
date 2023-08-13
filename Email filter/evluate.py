import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load your CSV data
df = pd.read_csv('data/data_produced.csv')
print(df.head())
# Count the occurrences of each unique value in the 'label' column
label_counts = df['Label'].value_counts()

# Print out the result
print(label_counts)
print(df.head(20))

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(''.join(df.Email))
plt.figure(figsize=(20, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

sns.countplot(x = df['Label'], data = df, order=['ham', 'spam'])
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#Load your data
data = pd.read_csv("data/data_produced.csv", encoding='utf-8')

common_words = get_top_n_words(data['Email'], 20)
for word, freq in common_words:
    print(word, freq)

