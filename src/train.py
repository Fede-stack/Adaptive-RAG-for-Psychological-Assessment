def train_zeroshot(cosine, type_embs, documents_retrieved):
    """zero-shot training with GPT-style model for BDI-II assessment scoring."""
    
    predictions_collection = []
    
    for doc_idx in range(len(docss)):
        #embedding generation based on model type
        doc_embeddings = {
            1: lambda: get_embedding(docss[doc_idx], model),
            2: lambda: get_embedding(docss[doc_idx], model, tokenizer),
        }.get(type_embs, lambda: np.array([get_embedding(doc) for doc in docss[doc_idx]]))()
        
        assessment_scores = []
        
        #process each BDI-II item (21 total)
        for item_idx in range(21):
            # Extract and aggregate relevant posts
            post_range = slice(item_idx * 4, (item_idx + 1) * 4)
            unique_posts = np.unique(
                list(itertools.chain.from_iterable(documents_retrieved[post_range]))
            )
            
            formatted_posts = '\n'.join(sorted(unique_posts))
            formatted_items = ''.join([
                f"{idx} {item}\n" 
                for idx, item in enumerate(bdi_items[item_idx])
            ])
            
            # openai example
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": ""
                    },
                    {
                        "role": "user", 
                        "content": f""
                    }
                ],
                temperature=0,
                max_tokens=3
            )
            
            assessment_scores.append(response.choices[0].message.content)
        
        #store predictions and display results
        current_predictions = np.array(assessment_scores)
        predictions_collection.append(current_predictions)
        
        print(f"Predictions: {current_predictions}")
        print(f"Ground Truth: {y[doc_idx]}")
        
        gc.collect()  
    
    return predictions_collection
