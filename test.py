import faiss
import numpy as np
import pandas as pd
import json

def create_faiss_index(master_df, embedding_col='Procedure_decription_embeddingvector'):
    """
    Builds a FAISS index from the master dataframe embeddings.
    """
    # Convert list of embeddings into a 2D float32 numpy array
    # We use np.stack to handle the series of lists/arrays
    master_vectors = np.stack(master_df[embedding_col].values).astype('float32')
    
    # Get the dimensionality of the embeddings
    dimension = master_vectors.shape[1]
    
    # Initialize a Flat L2 index (Euclidean distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add the vectors to the index
    index.add(master_vectors)
    
    return index



def search_similar_procedures(client_embedding, index, master_df, top_k=5):
    """
    Searches the FAISS index and returns results in JSONL format.
    """
    # Ensure query vector is a 2D float32 array
    query_vector = np.array([client_embedding]).astype('float32')
    
    # Search the index: d = distances, i = indices of the rows in master_df
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for rank, idx in enumerate(indices[0], start=1):
        # Retrieve original data from the dataframe using the index
        match_row = master_df.iloc[idx]
        
        entry = {
            "match_procedure_name": match_row['procedure name'],
            "matching_score": float(match_row['Score']),
            "match_procedure_description": match_row['Procedure_description'],
            "match_level": match_row['Level'],
            "rank": rank
        }
        results.append(entry)
    
    # Return as JSONL string
    return "\n".join([json.dumps(res) for res in results])
