import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA (GPU support) is available and enabled!")
    device = torch.device('cuda')
else:
    print("CUDA (GPU support) is not available. Using CPU.")
    device = torch.device('cpu')

# Load the FAISS index
faiss_index = faiss.read_index('./database/legislation_index.faiss')

# Load the filename mapping
with open('./database/metadata_mapping.json', 'r') as f:
    filenames = json.load(f)

# Load the sentence transformer model
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def retrieve_documents(query, k=5):
    query_embedding = sentence_model.encode(query, convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, idxs = faiss_index.search(query_embedding, k)
    return [(filenames[i], distances[0][j]) for j, i in enumerate(idxs[0])]

# Ask for model specification
model_spec = input("Specify model: 70b / 13b / 7b: ").strip()

# Mapping the input to the model directory names
model_directories = {
    '70b': 'Llama-2-70b-chat',
    '13b': 'Llama-2-13b-chat',
    '7b': 'Llama-2-7b-chat'
}

# Check if the specified model is valid
if model_spec in model_directories:
    model_directory = model_directories[model_spec]
    local_model_path = os.path.join("./models", model_directory)
    if not os.path.isdir(local_model_path):
        raise ValueError(f"The model directory {local_model_path} does not exist. Please download the model first.")
    
    print(f"Loading tokenizer from {local_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print("Tokenizer loaded successfully.")
    
    print(f"Loading model from {local_model_path}... This may take a while.")
    llama_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    
    # Move model to GPU if CUDA is available
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        llama_model = llama_model.to('cuda')
        print("Model is now on GPU.")
    else:
        print("Model will run on CPU.")
    
    print("Model loaded successfully.")
else:
    raise ValueError("Invalid model specification. Please specify one of '70b', '13b', or '7b'.")

if device == torch.device('cuda'):
    print("Moving model to GPU...")
    llama_model = llama_model.to(device)
    print("Model is now on GPU.")
else:
    print("Model will run on CPU.")

def generate_answer(query):
    with torch.no_grad():  # Disable gradient tracking
        retrieved_docs = retrieve_documents(query)
        context = " ".join([doc[0] for doc in retrieved_docs])  # Using only filenames for context
        input_text = query + " " + context
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output = llama_model.generate(input_ids, max_length=512)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Free up memory after generating the answer
    del input_ids
    del output
    torch.cuda.empty_cache()

    return answer

# Function to clear the console
def clear_console():
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)

# Infinite loop to continuously ask for input
while True:
    clear_console()  # Clear the console for a new question
    try:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = generate_answer(question)
        print(answer)
        input("Press Enter to ask another question...")  # Wait for user to press Enter
    except Exception as e:
        print(f"An error occurred: {e}")
        break  # Exit the loop if an error occurs
