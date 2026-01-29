import faiss
import numpy as np
import json

def create_cosine_index(master_df, embedding_col='Procedure_decription_embeddingvector'):
    """
    Builds a FAISS index optimized for Cosine Similarity.
    """
    # Convert to 2D float32 numpy array
    master_vectors = np.stack(master_df[embedding_col].values).astype('float32')
    
    # STEP 1: Normalize the master vectors to unit length
    faiss.normalize_L2(master_vectors)
    
    # STEP 2: Use IndexFlatIP (Inner Product)
    dimension = master_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(master_vectors)
    
    return index


def search_similar_procedures(client_embedding, index, master_df, top_k=5):
    """
    Searches the FAISS index using Cosine Similarity and returns JSONL.
    """
    # Convert query to 2D float32 array
    query_vector = np.array([client_embedding]).astype('float32')
    
    # STEP 3: Normalize the query vector
    faiss.normalize_L2(query_vector)
    
    # Search: 'similarities' will be the cosine similarity values
    similarities, indices = index.search(query_vector, top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], similarities[0]), start=1):
        match_row = master_df.iloc[idx]
        
        entry = {
            "match_procedure_name": match_row['procedure name'],
            "master_score": float(match_row['Score']),
            "cosine_similarity": round(float(score), 4), # 1.0 is perfect match
            "match_procedure_description": match_row['Procedure_description'],
            "match_level": match_row['Level'],
            "rank": rank
        }
        results.append(entry)
    
    return "\n".join([json.dumps(res) for res in results])
