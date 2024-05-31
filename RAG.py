import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import subprocess

# Function to load documents from folder
def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)
    return documents, filenames

# Function to get embeddings
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Load documents
folder_path = 'documents'
documents, filenames = load_documents_from_folder(folder_path)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Get embeddings for documents
document_embeddings = get_embeddings(documents, tokenizer, model)

# Create a FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(np.array(document_embeddings))

# Save the index and document data
faiss.write_index(index, 'document_index')
np.save('document_embeddings.npy', document_embeddings)
np.save('filenames.npy', filenames)

# Function to retrieve documents
def retrieve(query, index, embeddings, documents, filenames, tokenizer, model, top_k=5):
    query_embedding = get_embeddings([query], tokenizer, model)
    D, I = index.search(query_embedding, top_k)
    return [documents[i] for i in I[0]], [filenames[i] for i in I[0]]

# Function to generate response using Ollama llama3
def generate_response(query, retrieved_docs):
    prompt = f"Query: {query}\nDocuments:\n" + "\n".join(retrieved_docs)
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode('utf-8'),  # Encode the prompt as UTF-8
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode('utf-8').strip()  # Decode the output as UTF-8
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        print(f"Stdout: {e.stdout.decode('utf-8')}")
        print(f"Stderr: {e.stderr.decode('utf-8')}")
        return "Error generating response."

# Interactive loop to query the RAG system
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_docs, retrieved_filenames = retrieve(query, index, document_embeddings, documents, filenames, tokenizer, model)
    response = generate_response(query, retrieved_docs)
    print(f"Response:\n{response}\n")
