# Juice Chat Prototype

A simple LLM demo for questions related to the Juicebox protocol.

## Usage

```bash
# Install dependencies
pipenv install

# Add your OpenAI API key to embeddings.py and chat.py
vim embeddings.py
vim chat.py

# If needed, run embeddings.py to generate embeddings
pipenv run python embeddings.py

# Run chat
pipenv run python main.py
```
