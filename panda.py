import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("df_file.csv")

# 1. Average length of news articles
df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
print("1. Average word count:", df['word_count'].mean())

# 2. Label distribution
print("\n2. Label distribution:\n", df['Label'].value_counts())

# 3. Most frequent words in each label class
for label in df['Label'].unique():
    words = ' '.join(df[df['Label'] == label]['Text']).lower().split()
    most_common = Counter(words).most_common(10)
    print(f"\n3. Most common words for label {label}:\n", most_common)

# 4. Count duplicate articles
duplicates = df.duplicated(subset='Text').sum()
print("\n4. Duplicate articles count:", duplicates)

# 5. Missing values
print("\n5. Missing values:\n", df.isnull().sum())

# 6. Word count summary
print("\n6. Word count stats:\n", df['word_count'].describe())

# 7. Average word count by label
print("\n7. Average word count by label:\n", df.groupby('Label')['word_count'].mean())

# 8. Most common starting word for each label
df['first_word'] = df['Text'].apply(lambda x: str(x).split()[0].lower() if str(x).split() else "")
for label in df['Label'].unique():
    starts = df[df['Label'] == label]['first_word']
    print(f"\n8. Most common start word for label {label}:", starts.value_counts().idxmax())

# 9. Most frequent punctuation
punctuation_counts = Counter("".join(df['Text']).translate(str.maketrans('', '', string.ascii_letters + string.digits + " ")))
print("\n9. Most common punctuation marks:\n", punctuation_counts.most_common(5))

# 10. Stopword ratio
stop_words = set(stopwords.words('english'))
df['stopword_ratio'] = df['Text'].apply(
    lambda x: len([word for word in str(x).lower().split() if word in stop_words]) / len(str(x).split()) if len(str(x).split()) > 0 else 0
)
print("\n10. Average stopword ratio:", df['stopword_ratio'].mean())
