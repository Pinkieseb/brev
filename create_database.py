from sentence_transformers import SentenceTransformer
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
import faiss
import json
from pathlib import Path
import torch

if torch.cuda.is_available():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to('cuda')
    print("Using GPU for Sentence Transformers")
else:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    print("Using CPU for Sentence Transformers")


def embed_document(file_path, is_victorian):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        segments = content.split('\n\n')  # Splitting by empty lines
        results = []
        for segment in segments:
            if segment.strip():  # Ignore empty segments
                embedding = model.encode(segment, convert_to_tensor=False)
                label = "Victorian" if is_victorian else "Commonwealth"
                results.append((embedding, os.path.basename(file_path), segment, label))
        return results
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def process_directory(directory, is_victorian):
    # Use pathlib to create a Path object
    base_path = Path(directory)
    # Use rglob to recursively search for .txt files
    file_paths = list(base_path.rglob('*.txt'))
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks to the executor for each file path
        future_to_file = {executor.submit(embed_document, str(fp), is_victorian): fp for fp in file_paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_paths), desc="Embedding Documents"):
            results = future.result()
            all_results.extend(results)
    return all_results

vic_results = process_directory('./data/victorian_legislation/', True)
cth_results = process_directory('./data/commonwealth_legislation/', False)

all_embeddings = [result[0] for result in vic_results + cth_results]
all_metadata = [{"filename": result[1], "content": result[2], "label": result[3]} for result in vic_results + cth_results]

import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    # Create a flat (L2) index
    cpu_index = faiss.IndexFlatL2(dimension)
    
    combined_embeddings = np.vstack(embeddings).astype('float32')
    cpu_index.add(combined_embeddings)  # Add vectors to the index

    # Write index to disk
    faiss.write_index(cpu_index, 'legislation_index.faiss')

    return cpu_index

# Create the index
faiss_index = create_faiss_index(all_embeddings)
faiss.write_index(faiss_index, 'legislation_index.faiss')

# Save the metadata
with open('metadata_mapping.json', 'w') as f:
    json.dump(all_metadata, f)

