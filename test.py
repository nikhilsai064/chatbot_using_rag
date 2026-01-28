import pandas as pd
import numpy as np
import faiss
import json
import ast


def convert_embedding_to_array(embedding):
    """
    Convert embedding from various formats to numpy array.
    
    Parameters:
    -----------
    embedding : str, list, np.ndarray, or object
        Embedding in any format
    
    Returns:
    --------
    np.ndarray : Numpy array of the embedding
    """
    # If it's already a numpy array
    if isinstance(embedding, np.ndarray):
        return embedding
    
    # If it's a string representation of a list
    if isinstance(embedding, str):
        try:
            # Try to evaluate the string as a Python literal
            embedding = ast.literal_eval(embedding)
        except:
            # If that fails, try json.loads
            try:
                embedding = json.loads(embedding)
            except:
                raise ValueError(f"Cannot parse embedding string: {embedding[:100]}...")
    
    # If it's a list, convert to numpy array
    if isinstance(embedding, list):
        return np.array(embedding, dtype='float32')
    
    # If it's some other object with tolist() method
    if hasattr(embedding, 'tolist'):
        return np.array(embedding.tolist(), dtype='float32')
    
    # Last resort - try direct conversion
    return np.array(embedding, dtype='float32')


def create_faiss_index(master_df):
    """
    Create and build FAISS index from master_df (one-time operation).
    This prepares the index for efficient similarity searches.
    
    Parameters:
    -----------
    master_df : pandas.DataFrame
        DataFrame with 'Procedure_decription_embeddingvector' column
    
    Returns:
    --------
    tuple : (index, master_embeddings, dimension)
        - index: FAISS index ready for searching
        - master_embeddings: Normalized numpy array of master embeddings
        - dimension: Embedding dimension
    """
    
    print("Converting master embeddings to numpy arrays...")
    
    # Convert all embeddings to numpy arrays
    embeddings_list = []
    for idx, emb in enumerate(master_df['Procedure_decription_embeddingvector']):
        try:
            emb_array = convert_embedding_to_array(emb)
            embeddings_list.append(emb_array)
        except Exception as e:
            print(f"Warning: Error converting embedding at index {idx}: {e}")
            raise
    
    # Stack all embeddings into a 2D array
    master_embeddings = np.vstack(embeddings_list).astype('float32')
    
    print(f"Master embeddings shape: {master_embeddings.shape}")
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(master_embeddings)
    
    # Get the dimension of embeddings
    dimension = master_embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    
    # Add master embeddings to the index
    index.add(master_embeddings)
    
    print(f"✓ FAISS index created successfully!")
    print(f"  - Total procedures indexed: {len(master_df)}")
    print(f"  - Embedding dimension: {dimension}")
    
    return index, master_embeddings, dimension


def search_similar_procedures(test_embedding, index, master_df, top_k=5):
    """
    Search for similar procedures using pre-built FAISS index.
    
    Parameters:
    -----------
    test_embedding : list, numpy array, string, or object
        Single embedding vector from test data
    index : faiss.Index
        Pre-built FAISS index from create_faiss_index()
    master_df : pandas.DataFrame
        Master DataFrame with procedure details
    top_k : int, default=5
        Number of top similar results to return
    
    Returns:
    --------
    list : List of dictionaries containing top_k matches
    """
    
    # Convert test embedding to numpy array
    test_embedding_array = convert_embedding_to_array(test_embedding)
    
    # Reshape to 2D array (1, dimension)
    if test_embedding_array.ndim == 1:
        test_embedding_array = test_embedding_array.reshape(1, -1)
    
    # Ensure float32 dtype
    test_embedding_array = test_embedding_array.astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(test_embedding_array)
    
    # Search for top_k similar vectors
    similarities, indices = index.search(test_embedding_array, top_k)
    
    # Prepare results
    results = []
    
    for rank, (master_idx, similarity) in enumerate(zip(indices[0], similarities[0]), start=1):
        result = {
            'match_procedure_name': master_df.iloc[master_idx]['procedure name'],
            'matching_score': float(master_df.iloc[master_idx]['Score']),
            'match_procedure_description': master_df.iloc[master_idx]['Procedure_description'],
            'match_level': master_df.iloc[master_idx]['Level'],
            'similarity': float(similarity),
            'rank': rank
        }
        results.append(result)
    
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

print("="*70)
print("FAISS Similarity Search for Medical Procedures")
print("="*70)

# Check data types
print(f"\nMaster embedding dtype: {master_df['Procedure_decription_embeddingvector'].dtype}")
print(f"Test embedding dtype: {test_df['client_procedure_description_embeddingvector'].dtype}")

# Check first embedding sample
print(f"\nSample master embedding type: {type(master_df.iloc[0]['Procedure_decription_embeddingvector'])}")
print(f"Sample test embedding type: {type(test_df.iloc[0]['client_procedure_description_embeddingvector'])}")

# STEP 1: Create FAISS index ONE TIME (do this once at the beginning)
print("\n" + "="*70)
print("STEP 1: Creating FAISS index from master_df...")
print("="*70)
faiss_index, master_embeddings, embedding_dim = create_faiss_index(master_df)
print()

# STEP 2: Now you can search multiple times without rebuilding the index

# Search for first test record
print("="*70)
print("STEP 2: Searching for first test record (iloc[0])...")
print("="*70)
first_test_embedding = test_df.iloc[0]['client_procedure_description_embeddingvector']
results_1 = search_similar_procedures(first_test_embedding, faiss_index, master_df, top_k=5)

print(f"\nTest Procedure: {test_df.iloc[0]['client_procedure_name']}")
print(f"Client: {test_df.iloc[0]['client_name']}")
print(f"Description: {test_df.iloc[0]['client_procedure_description']}")
print("\nTop 5 Matches:")
for match in results_1:
    print(f"  Rank {match['rank']}: {match['match_procedure_name']} "
          f"(Similarity: {match['similarity']:.4f}, Level: {match['match_level']})")
print()

# Search for all test records
print("="*70)
print("STEP 3: Searching for ALL test records...")
print("="*70)
all_results = []
for idx in range(len(test_df)):
    test_embedding = test_df.iloc[idx]['client_procedure_description_embeddingvector']
    results = search_similar_procedures(test_embedding, faiss_index, master_df, top_k=5)
    all_results.append(results)
    print(f"  ✓ Processed test record {idx + 1}/{len(test_df)}: {test_df.iloc[idx]['client_procedure_name']}")

print(f"\n✓ Completed! Processed {len(test_df)} test records")

# Display detailed results for first record
print("\n" + "="*70)
print("Detailed JSON results for first test record:")
print("="*70)
print(json.dumps(all_results[0], indent=2))

# Save all results to JSONL
with open('all_similarity_results.jsonl', 'w') as f:
    for test_results in all_results:
        json.dump(test_results, f)
        f.write('\n')

print("\n✓ All results saved to 'all_similarity_results.jsonl'")
