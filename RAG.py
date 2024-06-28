import os
import redis
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tqdm.auto import tqdm
import subprocess

# Constants
REDIS_HOST = ""
REDIS_PORT = 000
REDIS_PASSWORD = ""
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
INDEX_NAME = "doc:idx"
DOCUMENTS_FOLDER = 'documents'

# Initialize Redis connection
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
redis_client.ping()

# Initialize the embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to load documents from a folder
def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)
    return documents, filenames

# Function to generate and normalize embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    embeddings = normalize(embeddings)  # Normalize embeddings
    return embeddings

# Load documents
documents, filenames = load_documents_from_folder(DOCUMENTS_FOLDER)

# Generate embeddings
document_embeddings = get_embeddings(documents)

# Function to save documents and embeddings to Redis
def load_documents_to_redis(redis_client, documents, embeddings, filenames, key_prefix="doc", pipe_size=100):
    pipe = redis_client.pipeline(transaction=False)
    for i, (doc, embedding, filename) in enumerate(tqdm(zip(documents, embeddings, filenames), total=len(documents))):
        key = f"{key_prefix}:{i}"
        pipe.hset(key, mapping={
            "filename": filename,
            "full_text": doc,
            "text_embedding": embedding.astype(np.float32).tobytes()
        })
        if (i + 1) % pipe_size == 0:
            pipe.execute()
    pipe.execute()

# Function to create Redis index
def create_redis_index(redis_client, idxname=INDEX_NAME):
    try:
        redis_client.ft(idxname).dropindex()
    except:
        print("No existing index found, creating a new one")

    index_definition = IndexDefinition(
        prefix=["doc:"],
        index_type=IndexType.HASH,
    )

    redis_client.ft(idxname).create_index(
        fields=[
            TextField("filename"),
            TextField("full_text"),
            VectorField("text_embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": 384,
                "DISTANCE_METRIC": "COSINE",
            })
        ],
        definition=index_definition
    )

# Clear Redis database (optional)
redis_client.flushdb()

# Create index
create_redis_index(redis_client)

# Load documents and embeddings to Redis
load_documents_to_redis(redis_client, documents, document_embeddings, filenames)

# Function to perform vector search
def vector_search(redis_client, query, top_k=10):
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    query_embedding = normalize(query_embedding).astype(np.float32).tobytes()

    search_query = Query(f"*=>[KNN {top_k} @text_embedding $vector AS result_score]") \
        .return_fields("result_score", "filename", "full_text") \
        .dialect(2) \
        .sort_by("result_score", True)
    query_params = {"vector": query_embedding}
    search_result = redis_client.ft(INDEX_NAME).search(search_query, query_params=query_params)
    return search_result

# Function to interact with Ollama using subprocess
def generate_ollama_response(query, documents):
    prompt = f"Query: {query}\nDocuments:\n" + "\n".join(documents)
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

# Function to retrieve documents and generate a focused response using Redis vector search and Ollama
def retrieve_and_generate_response(query, redis_client, top_k=5, relevance_threshold=0.2):
    if query.lower() in ['hi', 'hey', 'hello', 'greetings']:
        return "Hello! How can I assist you today?"

    search_result = vector_search(redis_client, query, top_k)
    
    if not search_result.docs:
        return "I couldn't find any relevant information for your query."

    results = []
    for doc in search_result.docs:
        if float(doc.result_score) >= relevance_threshold:
            results.append(doc.full_text)

    if results:
        return generate_ollama_response(query, results)
    return "No relevant documents found."

# Main interaction loop to continuously process queries
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = retrieve_and_generate_response(query, redis_client)
    print(f"Response:\n{response}\n")
