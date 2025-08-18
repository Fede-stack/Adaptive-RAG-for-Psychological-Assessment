import numpy as np
import random
import torch
import itertools
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from scipy.stats import ks_2samp
from dadapy import Data
from dadapy._utils import utils as ut
from sentence_transformers import util, SentenceTransformer
import logging
import warnings
from typing import List, Tuple, Optional, Union

class AdaptiveRetrieval:
    """
    Adaptive retrieval system based on intrinsic dimensionality
    for BDI-II analysis of Reddit posts.
    """
    
    def __init__(self, 
                 dthr: float = DTHR, #Dthr is the decision threshold - percentile of the χ²₁ distribution- that determines when to stop expanding neighborhoods based on density constancy tests.
                 r: Union[str, float] = 'opt',
                 n_iter: int = 10,
                 cosine_metric: bool = True,
                 random_seed: int = 42,
                 verbose: bool = False):
        """
        Initialize the adaptive retrieval system.
        
        Args:
            dthr: Threshold for k* calculation
            r: Ratio parameter ('opt' for optimal or float value)
            n_iter: Number of iterations for convergence
            cosine_metric: Whether to use cosine distance (True) or dot product (False)
            random_seed: Seed for reproducibility
            verbose: Whether to print debug information
        """
        self.dthr = dthr
        self.r = r
        self.n_iter = n_iter
        self.cosine_metric = cosine_metric
        self.random_seed = random_seed
        self.verbose = verbose
        
        #Set seed for reproducibility
        self.set_seed(random_seed)
        
        #Initialize random number generator
        self.rng = np.random.default_rng(random_seed)
        
        #Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def set_seed(self, seed: int) -> None:
        """Set seed for all random generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def compute_intrinsic_dimension_and_kstar(self, 
                                            data: Data, 
                                            embeddings: np.ndarray,
                                            initial_id: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute intrinsic dimensionality and k* using binomial approach.
        
        Args:
            data: dadapy Data object
            embeddings: Array of embeddings
            initial_id: Optional initial ID
            
        Returns:
            Tuple of (ids, kstars) for all iterations
        """
        if initial_id is None:
            data.compute_id_2NN(algorithm='base')
        else:
            data.compute_distances()
            data.set_id(initial_id)
        
        # Initialize arrays to store results
        ids = np.zeros(self.n_iter)
        ids_err = np.zeros(self.n_iter)
        kstars = np.zeros((self.n_iter, data.N), dtype=int)
        log_likelihoods = np.zeros(self.n_iter)
        ks_stats = np.zeros(self.n_iter)
        p_values = np.zeros(self.n_iter)
        
        for i in range(self.n_iter):
            if self.verbose:
                self.logger.info(f"Iteration {i+1}/{self.n_iter}")
            
            # Compute kstar
            data.compute_kstar(self.dthr)
            
            # Set new ratio
            r_eff = min(0.95, 0.2032**(1./data.intrinsic_dim)) if self.r == 'opt' else self.r
            
            # Compute neighbourhoods shells from k_star
            rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
            rn = rk * r_eff
            n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
            
            # Compute id
            id_current = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
            
            # Compute id error
            id_err = ut._compute_binomial_cramerrao(id_current, data.kstar-1, r_eff, data.N)
            
            # Compute likelihood
            log_lik = ut.binomial_loglik(id_current, data.kstar - 1, n - 1, r_eff)
            
            # Model validation through KS test
            n_model = self.rng.binomial(data.kstar-1, r_eff**id_current, size=len(n))
            ks, pv = ks_2samp(n-1, n_model)
            
            #Set new id
            data.set_id(id_current)
            
            #Store results
            ids[i] = id_current
            ids_err[i] = id_err
            kstars[i] = data.kstar
            log_likelihoods[i] = log_lik
            ks_stats[i] = ks
            p_values[i] = pv
        
        #Update final data attributes
        data.intrinsic_dim = id_current
        data.intrinsic_dim_err = id_err
        data.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())
        
        if self.verbose:
            self.logger.info(f"Final intrinsic dimension: {data.intrinsic_dim:.3f}")
        
        return ids, kstars[-1, :]
    
    def find_k_neighbors(self, 
                        embeddings: np.ndarray, 
                        target_index: int, 
                        k: int) -> List[int]:
        """
        Find k nearest neighbors for a target embedding.
        
        Args:
            embeddings: Array of embeddings
            target_index: Index of target embedding
            k: Number of neighbors to find
            
        Returns:
            List of indices of k nearest neighbors
        """
        target_embedding = embeddings[target_index]
        
        if self.cosine_metric:
            #Compute cosine distance
            all_distances = np.array([distance.cosine(target_embedding, emb) 
                                    for emb in embeddings])
            #Sort by ascending order (smaller distance = higher similarity)
            nearest_indices = np.argsort(all_distances)[1:k+1]
        else:
            # Compute dot product
            all_scores = util.dot_score(target_embedding, embeddings)[0].cpu().tolist()
            # ort by descending order (larger score = higher similarity)
            nearest_indices = np.argsort(all_scores)[::-1][1:k+1]
        
        return nearest_indices.tolist()
    
    def retrieve_documents(self, 
                          item_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray,
                          doc_texts: List[str]) -> Tuple[List[str], dict]:
        """
        Main method for document retrieval.
        
        Args:
            item_embedding: Embedding of BDI item
            doc_embeddings: Embeddings of documents
            doc_texts: Original document texts
            
        Returns:
            Tuple of (retrieved documents, metadata)
        """
        try:

            combined_embeddings = np.concatenate([
                item_embedding.reshape(1, -1), 
                doc_embeddings
            ])
            
            data = Data(combined_embeddings)
            
            #compute intrinsic dimensionality and k*
            ids, kstars = self.compute_intrinsic_dimension_and_kstar(data, doc_embeddings)
            
            #find neighbors for the item (index 0)
            k_optimal = int(kstars[0])
            retrieved_indices = self.find_k_neighbors(
                combined_embeddings, 
                target_index=0, 
                k=k_optimal
            )
            
            #convert from combined indices to document indices (subtract 1)
            doc_indices = [idx - 1 for idx in retrieved_indices if idx > 0 and idx-1 < len(doc_texts)]
            
            #get actual documents
            retrieved_docs = [doc_texts[idx] for idx in doc_indices]
            
            #metadata
            metadata = {
                'intrinsic_dimension': data.intrinsic_dim,
                'k_optimal': k_optimal,
                'n_retrieved': len(retrieved_docs),
                'convergence_ids': ids,
                'doc_indices': doc_indices
            }
            
            return retrieved_docs, metadata
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in document retrieval: {e}")
            #fallback to simple similarity
            fallback_docs = self._fallback_retrieval(item_embedding, doc_embeddings, doc_texts)
            return fallback_docs, {'fallback': True}
    
    def _fallback_retrieval(self, 
                           item_embedding: np.ndarray, 
                           doc_embeddings: np.ndarray,
                           doc_texts: List[str],
                           k: int = 10) -> List[str]:
        """Fallback to simple cosine similarity in case of error."""
        similarities = np.array([
            1 - distance.cosine(item_embedding, doc_emb) 
            for doc_emb in doc_embeddings
        ])
        top_indices = np.argsort(similarities)[::-1][:min(k, len(doc_texts))]
        return [doc_texts[idx] for idx in top_indices]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
model = SentenceTransformer(MODEL_ST_ID)

def get_embedding(text, model):
    return model.encode(text)

#get BDI-II item embeddings
items_embs = get_embedding(sentences_bdi, model)
# sentences_bdi is defined in BDI-II/bdi_choices.py

#initialize adaptive retrieval system
adaptive_retriever = AdaptiveRetrieval(
    dthr=6.67,
    r='opt',
    n_iter=10,
    cosine_metric=True,
    random_seed=42,
    verbose=False
)

cosine = True
warnings.filterwarnings('ignore')


docs_retrieved = []
error_indices = []

#########################################################
# docss is a list of posts for each user. docss will have
# shape (number_of_user_in_dataset, number_of_all_posts)
#########################################################

for j in range(len(docss)):

    doc_embeddings = get_embedding(docss[j], model) 
    
    documents_retrieved = []
    print(f"Processing user {j}")
    

    for i, item in enumerate(items_embs):
        try:

            retrieved_docs, metadata = adaptive_retriever.retrieve_documents(
                item, doc_embeddings, docss[j]
            )
            documents_retrieved.append(retrieved_docs)
            
        except ValueError as e:
            if "array must not contain infs or NaNs" in str(e):
                error_indices.append((i, j))
                documents_retrieved.append([])  # Add empty list as fallback
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error for item {i}, user {j}: {e}")
            documents_retrieved.append([])  # Add empty list as fallback
    
    docs_retrieved.append(documents_retrieved)
