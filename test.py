import pandas as pd
import numpy as np
import faiss
import json


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
    
    # Convert embedding to numpy array
    # Handle both list and already-numpy-array formats
    embeddings_list = []
    for emb in master_df['Procedure_decription_embeddingvector']:
        if isinstance(emb, str):
            # If stored as string, evaluate it
            emb = eval(emb)
        if isinstance(emb, list):
            embeddings_list.append(emb)
        else:
            # Already numpy array
            embeddings_list.append(emb.tolist() if hasattr(emb, 'tolist') else emb)
    
    master_embeddings = np.array(embeddings_list, dtype='float32')
    
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
    test_embedding : list or numpy array or string
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
    # Handle both list and string formats
    if isinstance(test_embedding, str):
        test_embedding = eval(test_embedding)
    
    if isinstance(test_embedding, list):
        test_embedding_array = np.array([test_embedding], dtype='float32')
    else:
        # Already numpy array
        test_embedding_array = np.array([test_embedding.tolist() if hasattr(test_embedding, 'tolist') else test_embedding], dtype='float32')
    
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

# STEP 1: Create FAISS index ONE TIME (do this once at the beginning)
print("Creating FAISS index from master_df...")
faiss_index, master_embeddings, embedding_dim = create_faiss_index(master_df)
print()

# STEP 2: Now you can search multiple times without rebuilding the index

# Search for first test record
print("Searching for first test record (iloc[0])...")
first_test_embedding = test_df.iloc[0]['client_procedure_description_embeddingvector']
results_1 = search_similar_procedures(first_test_embedding, faiss_index, master_df, top_k=5)

print(f"Test Procedure: {test_df.iloc[0]['client_procedure_name']}")
print("\nTop 5 Matches:")
for match in results_1:
    print(f"  Rank {match['rank']}: {match['match_procedure_name']} (Similarity: {match['similarity']:.4f})")
print()

# Search for second test record (if exists)
if len(test_df) > 1:
    print("Searching for second test record (iloc[1])...")
    second_test_embedding = test_df.iloc[1]['client_procedure_description_embeddingvector']
    results_2 = search_similar_procedures(second_test_embedding, faiss_index, master_df, top_k=5)
    
    print(f"Test Procedure: {test_df.iloc[1]['client_procedure_name']}")
    print("\nTop 5 Matches:")
    for match in results_2:
        print(f"  Rank {match['rank']}: {match['match_procedure_name']} (Similarity: {match['similarity']:.4f})")
    print()

# Search for all test records
print("Searching for ALL test records...")
all_results = []
for idx in range(len(test_df)):
    test_embedding = test_df.iloc[idx]['client_procedure_description_embeddingvector']
    results = search_similar_procedures(test_embedding, faiss_index, master_df, top_k=5)
    all_results.append(results)
    print(f"  ✓ Processed test record {idx + 1}/{len(test_df)}: {test_df.iloc[idx]['client_procedure_name']}")

print(f"\n✓ Completed! Processed {len(test_df)} test records")

# Display detailed results for first record
print("\n" + "="*70)
print("Detailed results for first test record:")
print("="*70)
print(json.dumps(all_results[0], indent=2))

# Save all results to JSONL
with open('all_similarity_results.jsonl', 'w') as f:
    for test_results in all_results:
        json.dump(test_results, f)
        f.write('\n')

print("\n✓ All results saved to 'all_similarity_results.jsonl'")
