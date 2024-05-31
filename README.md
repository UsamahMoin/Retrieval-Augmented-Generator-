# README.md

## Overview
This project is a simple implementation of a Retrieval-Augmented Generation (RAG) system using Hugging Face Transformers, FAISS for efficient similarity search, and the Ollama Llama3 model for response generation. The system loads text documents, processes them into embeddings, indexes them using FAISS, and allows interactive querying and response generation based on the retrieved documents.

## Requirements
- Python 3.7+
- Transformers (Hugging Face)
- PyTorch
- FAISS
- NumPy
- Subprocess module (part of Python standard library)
- Ollama CLI for Llama3 model

## Setup

### Install Dependencies
1. Clone the repository:
    ```bash
    git clone git@github.com:UsamahMoin/Retrieval-Augmented-Generator-.git
    cd Retrieval-Augmented-Generator--main
    ```

2. Install the required Python packages:
    ```bash
    pip install transformers torch faiss-cpu numpy
    ```

3. Ensure Ollama CLI is installed and configured:
    ```bash
    curl -sSfL https://ollama.com/download | sh
    ```
or Download Ollama from here -> https://ollama.com/

### Prepare Documents
1. Create a folder named `documents` in the project directory.
2. Place your text files (`.txt` format) inside the `documents` folder.

## Usage

### Running the Code
1. Execute the script:
    ```bash
    python RAG.py
    ```

2. The script will load the documents, create embeddings, and build a FAISS index.
3. Once the setup is complete, you will be prompted to enter your queries interactively.

### Querying
1. Enter a query when prompted.
2. The system will retrieve the top 5 most relevant documents and generate a response using the Llama3 model.
3. To exit the interactive loop, type `exit`.

## Functions

### `load_documents_from_folder(folder_path)`
- Loads text documents from the specified folder.
- Returns:
  - `documents`: List of document contents.
  - `filenames`: List of document filenames.

### `get_embeddings(texts, tokenizer, model)`
- Converts a list of texts into embeddings using the specified tokenizer and model.
- Returns:
  - `embeddings`: NumPy array of embeddings.

### `retrieve(query, index, embeddings, documents, filenames, tokenizer, model, top_k=5)`
- Retrieves the top `top_k` documents most relevant to the query.
- Returns:
  - List of retrieved document contents.
  - List of corresponding filenames.

### `generate_response(query, retrieved_docs)`
- Generates a response to the query based on the retrieved documents using the Llama3 model.
- Returns:
  - Response string.

## Example
```bash
$ python main.py
Enter your query (or type 'exit' to quit): What is the process of photosynthesis?
Response:
Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy...
    ```

