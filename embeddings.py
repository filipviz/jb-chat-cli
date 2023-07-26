import re
import os
import tiktoken
import pandas as pd
import openai

openai.api_key = ""

# Initialize a directory for csv files if one doesn't exist
if not os.path.exists("processed"):
    os.mkdir("processed")

# Function to sanitize docusaurus markdown text
def sanitize_markdown(text):
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text) # Remove images
    text = re.sub(r':::', '', text, flags=re.DOTALL) # Remove docusaurus admonitions
    text = re.sub(r'---.*?---', '', text, flags=re.DOTALL) # Remove markdown head matter
    text = re.sub(r'#{1,6} ', '', text) # Remove markdown headers
    text = text.replace('\n', ' ') # Remove newlines
    text = re.sub(r' +', ' ', text) # Replace multi-spaces with one space
    return text

texts=[]

# Function to walk through the provided directory, adding markdown file content to the texts list
def add_md_to_texts(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf8') as f:
                    text = f.read()
                texts.append((file_path, sanitize_markdown(text)))

# Read from relevant docs directories
add_md_to_texts("juice-docs/docs/user/")
# TODO: other directories

# Initialize pandas DataFrame, add texts
df = pd.DataFrame(texts, columns=['path', 'text'])
df.to_csv('processed/texts.csv')

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# New DataFrame column indicating the number of tokens in each text
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

max_tokens = 8191 # Max input tokens for text-embedding-ada-002

# Function to split text into embeddable chunks
def split_into_many(text, max_tokens=max_tokens):
    sentences = text.split(". ")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

shortened = []

for row in df.iterrows():
    if row[1]['text'] is None:
        continue

    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    else:
        shortened.append(row[1]['text'])

# Update DataFrame with embeddable (shortened) text and new token lengths
df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.to_csv("processed/shortened.csv")

# Generate embeddings with text-embedding-ada-002
# TODO: Handle rate limiting https://platform.openai.com/docs/guides/rate-limits (3,500 RPM; 350,000 TPM)
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding']) # pyright: ignore
df.to_csv("processed/embeddings.csv")
