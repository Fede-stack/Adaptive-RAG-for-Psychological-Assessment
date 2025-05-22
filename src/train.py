import gc
from typing import List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import openai
from dataclasses import dataclass
import itertools
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng()
import pandas as pd
from dadapy import Data
from dadapy._utils import utils as ut
import os

@dataclass
class Data:
    embeddings: np.ndarray

def train(cosine: bool, type_embs: int) -> List[np.ndarray]:
    """
    Train the model and generate predictions using document embeddings and LLMs.
    
    Args:
        cosine (bool): Whether to use cosine similarity
        type_embs (int): Type of embeddings to use (1: SentenceTransformers, 2: Transformers, other: Custom)
    
    Returns:
        List[np.ndarray]: List of LLMs predictions for all users
    """
    predictions_tot_gpt: List[np.ndarray] = []
    error_indices: List[Tuple[int, int]] = []
    
    #process each user
    for user_idx in tqdm(range(len(docss)), desc="Processing Users"):
        # Get document embeddings based on type
        doc_embeddings = _get_embeddings_by_type(docss[user_idx], type_embs)
        documents_retrieved = _process_item_embeddings(doc_embeddings, cosine, user_idx, error_indices)
        
        #Generate LLM predictions
        predictions_gpt = _generate_gpt_predictions(documents_retrieved)
        predictions_tot_gpt.append(predictions_gpt)
        
        #clean up memory
        gc.collect()
    
    return predictions_tot_gpt

def _get_embeddings_by_type(docs: List[str], type_embs: int) -> np.ndarray:
    """Get document embeddings based on the specified type."""
    if type_embs == 1:
        return get_embedding(docs, model)  # with SentenceTransformers
    elif type_embs == 2:
        return get_embedding(docs, model, tokenizer)  # with transformers
    return np.array([get_embedding(doc) for doc in docs])

def _process_item_embeddings(
    doc_embeddings: np.ndarray,
    cosine: bool,
    user_idx: int,
    error_indices: List[Tuple[int, int]]
) -> List[List[str]]:
    """Process embeddings for each item and retrieve relevant documents."""
    documents_retrieved = []
    
    for item_idx, item in enumerate(items_embs):
        try:
            # Concatenate item and document embeddings
            embs = np.concatenate((np.array(item).reshape(1, -1), doc_embeddings))
            data = Data(embs)
            
            # Get nearest neighbors
            ids, kstars = return_ids_kstar_binomial(
                data,
                doc_embeddings,
                initial_id=None,
                Dthr=6.67,
                r='opt',
                n_iter=10
            )
            nns = find_single_k_neighs(embs, 0, kstars[0], cosine)
            
            # Retrieve documents
            documents_retrieved.append(
                np.array(docss[user_idx])[np.array(nns) - 1].tolist()
            )
            
        except ValueError as e:
            if "array must not contain infs or NaNs" in str(e):
                error_indices.append((item_idx, user_idx))
            else:
                raise e
                
    return documents_retrieved

def _generate_gpt_predictions(documents_retrieved: List[List[str]]) -> np.ndarray:
    zero_shot_scores = []
    
    for i in range(21):
        # Process retrieved documents
        posts = np.unique(
            list(itertools.chain.from_iterable(
                documents_retrieved[(i*4):((i+1)*4)]
            ))
        ).tolist()
        posts.sort()
        posts_final = '\n '.join(posts)
        
        #Prepare content for LLM
        content = ''.join([
            f"{i} {item}\n "
            for i, item in enumerate(bdi_items[i])
        ])
        
        #Generate LLM response
        response = openai.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=MaxTokens
        )
        
        zero_shot_scores.append(response.choices[0].message.content)
    
    return np.array(zero_shot_scores_gpt)
