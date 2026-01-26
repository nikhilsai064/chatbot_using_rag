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

