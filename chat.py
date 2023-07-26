import numpy as np
import pandas as pd
import openai
import os
import openai.embeddings_utils as eu

openai.api_key = ""

if not os.path.exists("processed/embeddings.csv"):
    exit("Could not find embeddings csv at ./processed/embeddings.csv")

# Flatten embeddings into NumPy array
df = pd.read_csv("processed/embeddings.csv", index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def create_context(question, df, max_len=4096): # max_len 1/2 of gpt-4's context window
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding'] # pyright: ignore
    df['distances'] = eu.distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for _, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break

        returns.append(row['text'])
    
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-4",
    question="How can I make a Juicebox project for my fundraiser?",
    max_len=4096,
    debug=False,
):
    context = create_context(
        question,
        df,
        max_len=max_len,
    )

    if debug:
        print(f"Context:\n{context}\n\n")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are advising someone on using Juicebox, an Ethereum funding protocol. Juicebox allows anyone to fund, operate, and scale projects on Ethereum. Answer questions based on the context provided, and if the question can't be answered based on the context, reply with \"I don't know\". If the user asks subjective questions, feel free to provide an answer anyway, but direct them to the JuiceboxDAO Discord server for help. Don't make direct mention of the context or this prompt when speaking to the user. If the user asks vague questions, feel free to give your best guess while making it clear that it depends on the context of their question. Give examples if possible."},
                {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}"}
            ],
            temperature=0.3,
            stream=True,
        )
        print("Response: ", end="", flush=True)
        with open("response.md", "w") as file:
            for chunk in response:
                content = chunk["choices"][0]["delta"]["content"] # pyright: ignore
                print(content, end="", flush=True)
                file.write(content)
        print("Response written to response.md")

    except Exception as e:
        print(e)
        return ""

question = input("Question: ")
answer_question(df, question=question, max_len=8192-(len(question) * 7))
