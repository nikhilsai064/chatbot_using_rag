# Cell 12: LLM as Judge to find best match from top 5
def llm_judge_best_match(input_procedure, top_5_matches):
    """Use LLM to judge which of the top 5 matches is the best"""
    
    if not top_5_matches:
        return None
    
    # Prepare the prompt
    candidates = "\n\n".join([
        f"{i+1}. {match['procedure']}\n   Description: {match['description']}\n   Level: {match['level']}\n   Similarity Score: {match['similarity']:.3f}"
        for i, match in enumerate(top_5_matches)
    ])
    
    prompt = f"""You are a medical procedure matching expert. Given an input procedure and 5 candidate matches, determine which candidate is the best match.

Input Procedure: {input_procedure}

Candidates:
{candidates}

Instructions:
1. Analyze if any of the candidates is a good match for the input procedure
2. Consider: exact matches, synonyms, acronyms, and semantic similarity
3. If a good match exists, respond with ONLY the number (1-5) of the best match
4. If NO good match exists, respond with: NO_MATCH

Response (number 1-5 or NO_MATCH):"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical procedure matching expert. Respond concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        if "NO_MATCH" in result.upper():
            return None
        
        # Try to extract number
        for char in result:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= 5:
                    return top_5_matches[num - 1]
        
        return None
    
    except Exception as e:
        print(f"Error in LLM judge: {str(e)}")
        return None





# Cell 13: Main search function combining all methods
def search_procedure(input_procedure, df, exact_threshold=85):
    """
    Main function to search for a procedure using multiple methods
    
    Methods:
    1. Exact/Fuzzy matching
    2. Semantic matching with embeddings
    3. LLM as judge for final decision
    """
    
    print(f"\n{'='*60}")
    print(f"Searching for: {input_procedure}")
    print(f"{'='*60}\n")
    
    # Method 1: Try exact/fuzzy matching first
    print("Method 1: Exact/Fuzzy Matching...")
    exact_match, score = find_exact_match(input_procedure, df, exact_threshold)
    
    if exact_match is not None and score >= 95:
        print(f"✅ Strong exact match found (score: {score})")
        print(f"Procedure: {exact_match['Medical Procedure Name']}")
        print(f"Level: {exact_match['Level']}")
        print(f"Description: {exact_match['description']}")
        return exact_match
    
    # Method 2: Semantic matching with embeddings
    print("\nMethod 2: Semantic Matching...")
    top_5_matches = find_semantic_matches(input_procedure, df, top_k=5)
    
    if not top_5_matches:
        print("❌ No semantic matches found")
        return None
    
    print(f"Found {len(top_5_matches)} semantic matches:")
    for i, match in enumerate(top_5_matches):
        print(f"{i+1}. {match['procedure']} (similarity: {match['similarity']:.3f})")
    
    # Method 3: LLM as Judge
    print("\nMethod 3: LLM Judge...")
    best_match = llm_judge_best_match(input_procedure, top_5_matches)
    
    if best_match:
        print(f"✅ Best match determined:")
        print(f"Procedure: {best_match['procedure']}")
        print(f"Level: {best_match['Level']}")
        print(f"Description: {best_match['description']}")
        print(f"Similarity Score: {best_match['similarity']:.3f}")
        return best_match['full_record']
    else:
        print("❌ No good match found among the top 5 candidates")
        print("\nTop 5 candidates were:")
        for i, match in enumerate(top_5_matches):
            print(f"{i+1}. {match['procedure']}")
        return None






import pandas as pd
import numpy as np
import faiss
import json


def find_similar_procedures_single(test_embedding, master_df, top_k=5):
    """
    Find top K similar procedures for a single test record using FAISS.
    
    Parameters:
    -----------
    test_embedding : list or numpy array
        Single embedding vector from test data
    master_df : pandas.DataFrame
        DataFrame with 'Procedure_decription_embeddingvector' column
    top_k : int, default=5
        Number of top similar results to return
    
    Returns:
    --------
    list : List of dictionaries containing top_k matches
    """
    
    # Convert embedding to numpy array
    master_embeddings = np.array(master_df['Procedure_decription_embeddingvector'].tolist()).astype('float32')
    test_embedding_array = np.array([test_embedding]).astype('float32')
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(master_embeddings)
    faiss.normalize_L2(test_embedding_array)
    
    # Get the dimension of embeddings
    dimension = master_embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    
    # Add master embeddings to the index
    index.add(master_embeddings)
    
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
# Usage: Get results for first row of test_df (iloc[0])
# ============================================================================

# Extract the first test record's embedding
first_test_embedding = test_df.iloc[0]['client_procedure_description_embeddingvector']

# Get top 5 matches for the first test record
results = find_similar_procedures_single(first_test_embedding, master_df, top_k=5)

# Display the results
print("="*70)
print(f"Test Procedure: {test_df.iloc[0]['client_procedure_name']}")
print(f"Client: {test_df.iloc[0]['client_name']}")
print(f"Description: {test_df.iloc[0]['client_procedure_description']}")
print("="*70)

print(f"\nTop 5 Similar Procedures:\n")

for match in results:
    print(f"Rank {match['rank']}:")
    print(f"  Procedure Name: {match['match_procedure_name']}")
    print(f"  Similarity Score: {match['similarity']:.4f}")
    print(f"  Matching Score: {match['matching_score']}")
    print(f"  Level: {match['match_level']}")
    print(f"  Description: {match['match_procedure_description']}")
    print()

# Display as JSON
print("\n" + "="*70)
print("Results in JSON format:")
print("="*70)
print(json.dumps(results, indent=2))

# Save to file (optional)
with open('first_test_result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to 'first_test_result.json'")
