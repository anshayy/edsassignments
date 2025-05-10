import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('df_file.csv')  # Adjust path if needed

# Extract columns
texts = df['Text'].values
labels = df['Label'].values

# 1. Count how many samples belong to each unique label
unique_labels, label_counts = np.unique(labels, return_counts=True)
print("Label counts:", dict(zip(unique_labels, label_counts)))

# 2. Average length (in characters) of all text entries
avg_text_length = np.mean([len(t) for t in texts])
print("Average text length (characters):", avg_text_length)

# 3. Standard deviation of text lengths
std_text_length = np.std([len(t) for t in texts])
print("Standard deviation of text lengths:", std_text_length)

# 4. Boolean mask for samples labeled as class 0
mask_class_0 = labels == 0
print("Mask for class 0 (first 5):", mask_class_0[:5])

# 5. Maximum number of words in any single text entry
max_words_in_text = max([len(t.split()) for t in texts])
print("Max words in a single text:", max_words_in_text)

# 6. Indices of the top 5 longest text samples
text_lengths = np.array([len(t) for t in texts])
top_5_indices = text_lengths.argsort()[-5:][::-1]
print("Top 5 longest text indices:", top_5_indices)

# 7. Count of texts with more than 100 characters
count_texts_gt_100 = np.sum(text_lengths > 100)
print("Texts with more than 100 characters:", count_texts_gt_100)

# 8. Mean number of words per text
mean_words_per_text = np.mean([len(t.split()) for t in texts])
print("Mean words per text:", mean_words_per_text)

# 9. Normalized array of text lengths
normalized_lengths = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min())
print("Normalized lengths (first 5):", normalized_lengths[:5])

# 10. Check for duplicate text entries
has_duplicates = len(np.unique(texts)) != len(texts)
print("Any duplicates in text?", has_duplicates)


