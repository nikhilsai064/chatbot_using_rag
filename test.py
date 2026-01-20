# Cell 1: Install required packages
# Run this first if packages are not installed
"""
!pip install openai pandas openpyxl python-dotenv fuzzywuzzy python-Levenshtein scikit-learn numpy
"""

# Cell 2: Import libraries and load environment variables
import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL")
API_KEY = os.getenv("API_KEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=DEPLOYMENT_URL,
    api_key=API_KEY,
    api_version="2024-02-15-preview"
)

MODEL_NAME = "gpt-4.1-mini"  # Your deployment name

print("✅ Libraries imported and Azure OpenAI client initialized")

# Cell 3: Load master Excel file
# Replace 'master_procedures.xlsx' with your actual file name
master_file = 'master_procedures.xlsx'
df = pd.read_excel(master_file)

print(f"Loaded {len(df)} procedures from master file")
print("\nFirst few rows:")
print(df.head())

# Cell 4: Function to generate procedure description using Azure OpenAI
def generate_procedure_description(procedure_name, level):
    """Generate a medical procedure description using Azure OpenAI"""
    try:
        prompt = f"""Generate a concise, professional medical description for the following procedure:
        
Procedure: {procedure_name}
Complexity Level: {level}/5

Provide a 2-3 sentence description that includes:
- What the procedure is
- Its primary purpose
- General complexity indication

Keep it factual and clinical."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medical documentation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating description for {procedure_name}: {str(e)}")
        return None

# Test with first procedure
test_desc = generate_procedure_description(df.iloc[0]['medical procedure name'], df.iloc[0]['level'])
print(f"\nTest description generated:\n{test_desc}")

# Cell 5: Generate descriptions for all procedures
print("Generating descriptions for all procedures...")

descriptions = []
for idx, row in df.iterrows():
    desc = generate_procedure_description(row['medical procedure name'], row['level'])
    descriptions.append(desc)
    
    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(df)} procedures")

df['description'] = descriptions
print("\n✅ All descriptions generated")
print(df[['medical procedure name', 'description']].head())

# Cell 6: Function to generate embeddings using Azure OpenAI
def generate_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding for given text"""
    try:
        # Clean the text
        text = text.replace("\n", " ").strip()
        
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        return response.data[0].embedding
    
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

# Test embedding generation
test_embedding = generate_embedding(df.iloc[0]['description'])
print(f"Test embedding generated. Dimension: {len(test_embedding) if test_embedding else 0}")

# Cell 7: Generate embeddings for all procedure descriptions
print("Generating embeddings for all procedures...")

embeddings = []
for idx, row in df.iterrows():
    embedding = generate_embedding(row['description'])
    embeddings.append(embedding)
    
    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(df)} procedures")

df['embedding'] = embeddings
print("\n✅ All embeddings generated")

# Cell 8: Save master dictionary to Excel
output_file = 'master_dict.xlsx'

# Convert embeddings to string for Excel storage
df['embedding_str'] = df['embedding'].apply(lambda x: json.dumps(x) if x else None)

# Save without the list column (save string version instead)
df_to_save = df.drop(columns=['embedding'])
df_to_save.to_excel(output_file, index=False)

print(f"✅ Master dictionary saved to {output_file}")
print(f"Columns saved: {list(df_to_save.columns)}")

# Cell 9: Load master dictionary for searching
def load_master_dict(file_path='master_dict.xlsx'):
    """Load master dictionary and parse embeddings"""
    df = pd.read_excel(file_path)
    
    # Parse embeddings from string back to list
    df['embedding'] = df['embedding_str'].apply(lambda x: json.loads(x) if pd.notna(x) else None)
    
    return df

# Load the saved file
master_df = load_master_dict(output_file)
print(f"✅ Loaded {len(master_df)} procedures from master dictionary")

# Cell 10: Function for exact name matching using fuzzy matching
def find_exact_match(input_procedure, master_df, threshold=85):
    """Find exact or close matches using fuzzy string matching"""
    
    best_match = None
    best_score = 0
    
    for idx, row in master_df.iterrows():
        # Use token sort ratio for better matching
        score = fuzz.token_sort_ratio(
            input_procedure.lower(), 
            row['medical procedure name'].lower()
        )
        
        if score > best_score:
            best_score = score
            best_match = row
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

# Test exact matching
test_input = master_df.iloc[0]['medical procedure name']
match, score = find_exact_match(test_input, master_df)
print(f"Test input: {test_input}")
print(f"Match found: {match['medical procedure name'] if match is not None else 'None'}")
print(f"Score: {score}")

# Cell 11: Function for semantic/synonym matching using embeddings
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors

def find_semantic_matches(input_procedure, master_df, top_k=5, method='cosine'):
    """
    Find semantically similar procedures using embeddings
    
    Parameters:
    - input_procedure: str, the procedure to search for
    - master_df: DataFrame, the master dictionary
    - top_k: int, number of top matches to return
    - method: str, similarity/distance metric to use
        Options: 'cosine', 'euclidean', 'manhattan', 'dot_product', 'combined', 
                 'knn_cosine', 'knn_euclidean', 'knn_manhattan'
    """
    
    # Generate description for input
    input_desc = generate_procedure_description(input_procedure, level="Unknown")
    
    if not input_desc:
        return []
    
    # Generate embedding for input
    input_embedding = generate_embedding(input_desc)
    
    if not input_embedding:
        return []
    
    # Convert to numpy arrays for faster computation
    input_vec = np.array(input_embedding).reshape(1, -1)
    
    # Filter out rows with valid embeddings
    valid_embeddings = []
    valid_indices = []
    for idx, row in master_df.iterrows():
        if row['embedding'] is not None:
            valid_embeddings.append(row['embedding'])
            valid_indices.append(idx)
    
    if not valid_embeddings:
        return []
    
    embeddings_matrix = np.array(valid_embeddings)
    
    # KNN-based methods
    if method.startswith('knn_'):
        knn_metric = method.split('_')[1]  # Extract metric (cosine, euclidean, manhattan)
        
        # Map to sklearn metric names
        metric_map = {
            'cosine': 'cosine',
            'euclidean': 'euclidean',
            'manhattan': 'manhattan'
        }
        
        sklearn_metric = metric_map.get(knn_metric, 'cosine')
        
        # Create and fit KNN model
        knn = NearestNeighbors(
            n_neighbors=min(top_k, len(valid_embeddings)),
            metric=sklearn_metric,
            algorithm='brute'  # brute for small datasets, 'ball_tree' or 'kd_tree' for large
        )
        knn.fit(embeddings_matrix)
        
        # Find nearest neighbors
        distances, indices = knn.kneighbors(input_vec)
        
        # Convert distances to similarity scores
        # For cosine: distance is already 0-2, convert to similarity 0-1
        # For euclidean/manhattan: use inverse distance
        similarities = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            actual_idx = valid_indices[idx]
            
            if sklearn_metric == 'cosine':
                # Cosine distance is 1 - cosine_similarity
                # So similarity = 1 - distance
                score = 1 - dist
            else:
                # For euclidean and manhattan, convert distance to similarity
                score = 1 / (1 + dist)
            
            similarities.append((actual_idx, score))
    
    else:
        # Original non-KNN methods
        similarities = []
        
        for idx, row in master_df.iterrows():
            if row['embedding'] is not None:
                candidate_vec = np.array(row['embedding']).reshape(1, -1)
                
                if method == 'cosine':
                    # Cosine similarity (higher is better)
                    score = cosine_similarity(input_vec, candidate_vec)[0][0]
                    
                elif method == 'euclidean':
                    # Euclidean distance (lower is better, so we negate)
                    distance = euclidean(input_vec[0], candidate_vec[0])
                    # Convert to similarity score (inverse distance)
                    score = 1 / (1 + distance)
                    
                elif method == 'manhattan':
                    # Manhattan distance (lower is better, so we negate)
                    distance = cityblock(input_vec[0], candidate_vec[0])
                    # Convert to similarity score
                    score = 1 / (1 + distance)
                    
                elif method == 'dot_product':
                    # Dot product (higher is better)
                    score = np.dot(input_vec[0], candidate_vec[0])
                    # Normalize to 0-1 range
                    score = (score + 1) / 2
                    
                elif method == 'combined':
                    # Combined approach: average of multiple metrics
                    cos_sim = cosine_similarity(input_vec, candidate_vec)[0][0]
                    
                    euc_dist = euclidean(input_vec[0], candidate_vec[0])
                    euc_sim = 1 / (1 + euc_dist)
                    
                    dot_prod = np.dot(input_vec[0], candidate_vec[0])
                    dot_sim = (dot_prod + 1) / 2
                    
                    # Weighted average (cosine weighted more heavily)
                    score = 0.5 * cos_sim + 0.3 * euc_sim + 0.2 * dot_sim
                    
                else:
                    # Default to cosine
                    score = cosine_similarity(input_vec, candidate_vec)[0][0]
                
                similarities.append((idx, score))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:top_k]
    
    # Return top matching procedures
    results = []
    for idx, sim_score in similarities:
        results.append({
            'procedure': master_df.iloc[idx]['medical procedure name'],
            'description': master_df.iloc[idx]['description'],
            'level': master_df.iloc[idx]['level'],
            'similarity': sim_score,
            'method': method,
            'full_record': master_df.iloc[idx]
        })
    
    return results

# Cell 11.1: Compare different similarity methods (including KNN)
def compare_similarity_methods(input_procedure, master_df, top_k=5):
    """Compare results from different similarity methods including KNN"""
    
    methods = [
        'cosine', 'euclidean', 'manhattan', 'dot_product', 'combined',
        'knn_cosine', 'knn_euclidean', 'knn_manhattan'
    ]
    
    print(f"\nComparing similarity methods for: {input_procedure}")
    print("="*80)
    
    all_results = {}
    
    for method in methods:
        print(f"\n{method.upper()} Method:")
        print("-"*80)
        
        results = find_semantic_matches(input_procedure, master_df, top_k, method)
        all_results[method] = results
        
        for i, match in enumerate(results):
            print(f"{i+1}. {match['procedure']:<40} | Score: {match['similarity']:.4f}")
    
    return all_results

# Cell 11.2: Advanced KNN with custom parameters
def find_knn_matches_advanced(input_procedure, master_df, top_k=5, 
                               metric='cosine', algorithm='auto', weights='uniform'):
    """
    Advanced KNN search with more control over parameters
    
    Parameters:
    - input_procedure: str, the procedure to search for
    - master_df: DataFrame, the master dictionary
    - top_k: int, number of neighbors to find
    - metric: str, distance metric ('cosine', 'euclidean', 'manhattan', 'minkowski', 'chebyshev')
    - algorithm: str, algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute')
    - weights: str, weight function ('uniform' or 'distance')
    """
    
    # Generate description and embedding for input
    input_desc = generate_procedure_description(input_procedure, level="Unknown")
    if not input_desc:
        return []
    
    input_embedding = generate_embedding(input_desc)
    if not input_embedding:
        return []
    
    input_vec = np.array(input_embedding).reshape(1, -1)
    
    # Prepare embeddings matrix
    valid_embeddings = []
    valid_indices = []
    for idx, row in master_df.iterrows():
        if row['embedding'] is not None:
            valid_embeddings.append(row['embedding'])
            valid_indices.append(idx)
    
    if not valid_embeddings:
        return []
    
    embeddings_matrix = np.array(valid_embeddings)
    
    # Create KNN model
    knn = NearestNeighbors(
        n_neighbors=min(top_k, len(valid_embeddings)),
        metric=metric,
        algorithm=algorithm
    )
    knn.fit(embeddings_matrix)
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(input_vec)
    
    # Apply weights if specified
    if weights == 'distance':
        # Weight by inverse distance
        weighted_scores = 1 / (1 + distances[0])
    else:
        # Uniform weights (just convert distance to similarity)
        if metric == 'cosine':
            weighted_scores = 1 - distances[0]
        else:
            weighted_scores = 1 / (1 + distances[0])
    
    # Prepare results
    results = []
    for i, (dist, idx, score) in enumerate(zip(distances[0], indices[0], weighted_scores)):
        actual_idx = valid_indices[idx]
        results.append({
            'procedure': master_df.iloc[actual_idx]['medical procedure name'],
            'description': master_df.iloc[actual_idx]['description'],
            'level': master_df.iloc[actual_idx]['level'],
            'similarity': score,
            'distance': dist,
            'method': f'knn_{metric}',
            'algorithm': algorithm,
            'weights': weights,
            'full_record': master_df.iloc[actual_idx]
        })
    
    return results

# Test different KNN configurations (optional - uncomment to run)
# test_procedure = master_df.iloc[0]['medical procedure name']
# 
# print("\n" + "="*80)
# print("Testing KNN with different configurations:")
# print("="*80)
# 
# configs = [
#     {'metric': 'cosine', 'algorithm': 'brute', 'weights': 'uniform'},
#     {'metric': 'euclidean', 'algorithm': 'auto', 'weights': 'distance'},
#     {'metric': 'manhattan', 'algorithm': 'brute', 'weights': 'distance'},
# ]
# 
# for config in configs:
#     print(f"\nConfig: {config}")
#     results = find_knn_matches_advanced(test_procedure, master_df, top_k=3, **config)
#     for i, match in enumerate(results):
#         print(f"{i+1}. {match['procedure']} | Score: {match['similarity']:.4f} | Dist: {match['distance']:.4f}")

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
            model=MODEL_NAME,
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
def search_procedure(input_procedure, master_df, exact_threshold=85, similarity_method='cosine'):
    """
    Main function to search for a procedure using multiple methods
    
    Parameters:
    - input_procedure: str, the procedure to search for
    - master_df: DataFrame, the master dictionary
    - exact_threshold: int, threshold for fuzzy matching (0-100)
    - similarity_method: str, method for semantic matching
        Options: 'cosine', 'euclidean', 'manhattan', 'dot_product', 'combined',
                 'knn_cosine', 'knn_euclidean', 'knn_manhattan'
    
    Methods:
    1. Exact/Fuzzy matching
    2. Semantic matching with embeddings (including KNN)
    3. LLM as judge for final decision
    """
    
    print(f"\n{'='*60}")
    print(f"Searching for: {input_procedure}")
    print(f"Similarity Method: {similarity_method}")
    print(f"{'='*60}\n")
    
    # Method 1: Try exact/fuzzy matching first
    print("Method 1: Exact/Fuzzy Matching...")
    exact_match, score = find_exact_match(input_procedure, master_df, exact_threshold)
    
    if exact_match is not None and score >= 95:
        print(f"✅ Strong exact match found (score: {score})")
        print(f"Procedure: {exact_match['medical procedure name']}")
        print(f"Level: {exact_match['level']}")
        print(f"Description: {exact_match['description']}")
        return exact_match
    
    # Method 2: Semantic matching with embeddings
    print(f"\nMethod 2: Semantic Matching ({similarity_method})...")
    top_5_matches = find_semantic_matches(input_procedure, master_df, top_k=5, method=similarity_method)
    
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
        print(f"Level: {best_match['level']}")
        print(f"Description: {best_match['description']}")
        print(f"Similarity Score: {best_match['similarity']:.3f}")
        return best_match['full_record']
    else:
        print("❌ No good match found among the top 5 candidates")
        print("\nTop 5 candidates were:")
        for i, match in enumerate(top_5_matches):
            print(f"{i+1}. {match['procedure']}")
        return None

# Cell 14: Test cases
print("Running test cases...\n")

# Test Case 1: Exact name
test1 = master_df.iloc[0]['medical procedure name']
result1 = search_procedure(test1, master_df, similarity_method='knn_cosine')

# Test Case 2: Synonym with different KNN methods
test2 = "heart surgery"  # Replace with actual synonym from your domain
print("\n" + "="*80)
print("Testing with different KNN methods:")
print("="*80)

for method in ['knn_cosine', 'knn_euclidean', 'knn_manhattan', 'combined']:
    print(f"\n--- Using {method.upper()} ---")
    result2 = search_procedure(test2, master_df, similarity_method=method)

# Test Case 3: Acronym
test3 = "CABG"  # Replace with actual acronym from your domain
result3 = search_procedure(test3, master_df, similarity_method='knn_cosine')

# Test Case 4: Advanced KNN comparison
print("\n" + "="*80)
print("Advanced KNN Configuration Comparison:")
print("="*80)

test4 = "cardiac procedure"  # Replace with your test
configs = [
    {'metric': 'cosine', 'algorithm': 'brute', 'weights': 'uniform'},
    {'metric': 'euclidean', 'algorithm': 'auto', 'weights': 'distance'},
    {'metric': 'manhattan', 'algorithm': 'brute', 'weights': 'distance'},
]

for config in configs:
    print(f"\nKNN Config: {config}")
    results = find_knn_matches_advanced(test4, master_df, top_k=3, **config)
    for i, match in enumerate(results):
        print(f"{i+1}. {match['procedure']:<40} | Score: {match['similarity']:.4f}")

# Cell 15: Interactive search function
def interactive_search(default_method='knn_cosine'):
    """
    Interactive function to search for procedures
    
    Parameters:
    - default_method: str, default similarity method to use
    """
    
    print(f"\nDefault similarity method: {default_method}")
    print("\nAvailable methods:")
    print("  Standard: cosine, euclidean, manhattan, dot_product, combined")
    print("  KNN-based: knn_cosine, knn_euclidean, knn_manhattan")
    
    while True:
        print("\n" + "="*60)
        user_input = input("Enter procedure name to search (or 'quit' to exit): ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting search...")
            break
        
        if not user_input:
            print("Please enter a valid procedure name")
            continue
        
        # Ask if user wants to change method
        change_method = input(f"Use {default_method} method? (y/n, default=y): ").strip().lower()
        
        if change_method == 'n':
            method = input("Enter method: ").strip().lower()
            valid_methods = ['cosine', 'euclidean', 'manhattan', 'dot_product', 'combined',
                           'knn_cosine', 'knn_euclidean', 'knn_manhattan']
            if method not in valid_methods:
                print(f"Invalid method. Using {default_method}")
                method = default_method
        else:
            method = default_method
        
        result = search_procedure(user_input, master_df, similarity_method=method)
        
        if result is not None:
            print("\n" + "="*60)
            print("FULL RECORD FOUND:")
            print("="*60)
            for col in result.index:
                if col not in ['embedding', 'embedding_str']:
                    print(f"{col}: {result[col]}")
        else:
            print("\n⚠️ No matching procedure found in the master dictionary")

# Uncomment to run interactive search with KNN as default
# interactive_search(default_method='knn_cosine')
