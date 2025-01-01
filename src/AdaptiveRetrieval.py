import itertools
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng()
from dadapy import Data
from dadapy._utils import utils as ut
import os
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer, util

import numpy as np
from scipy.spatial import distance
from scipy.stats import ks_2samp
import torch.nn.functional as F
from sentence_transformers import util

class AdaptiveRetrieval:
    def __init__(self, data, embeddings):
        """
        Initialize the EmbeddingAnalyzer with data and embeddings.
        
        Args:
            data: Data object containing required methods (compute_id_2NN, compute_distances, set_id)
            embeddings: numpy array of embeddings
        """
        self.data = data
        self.embeddings = embeddings
        self.rng = np.random.default_rng()
        self.N = len(embeddings)
        self.distances = None
        self.intrinsic_dim = None
        self.intrinsic_dim_err = None
        self.intrinsic_dim_scale = None
        self.kstar = None

    def compute_kstar_binomial_id(self, initial_id=None, Dthr=23, r='opt', n_iter=10):
        """
        Compute k* and binomial ID estimation.
        
        Args:
            initial_id: Initial intrinsic dimension estimate
            Dthr: Distance threshold
            r: Ratio parameter ('opt' for automatic or float value)
            n_iter: Number of iterations
            
        Returns:
            tuple: (ids, kstars) for the final iteration
        """
        if initial_id is None:
            self.data.compute_id_2NN(algorithm='base')
        else:
            self.data.compute_distances()
            self.data.set_id(initial_id)

        ids = np.zeros(n_iter)
        ids_err = np.zeros(n_iter)
        kstars = np.zeros((n_iter, self.N), dtype=int)
        log_likelihoods = np.zeros(n_iter)
        ks_stats = np.zeros(n_iter)
        p_values = np.zeros(n_iter)

        for i in range(n_iter):
            # Compute kstar
            self.data.compute_kstar(Dthr)
            
            # Set new ratio
            r_eff = min(0.95, 0.2032**(1./self.data.intrinsic_dim)) if r == 'opt' else r
            
            # Compute neighbourhood shells from k_star
            rk = np.array([dd[self.data.kstar[j]] for j, dd in enumerate(self.data.distances)])
            rn = rk * r_eff
            n = np.sum([dd < rn[j] for j, dd in enumerate(self.data.distances)], axis=1)
            
            # Compute ID
            id = np.log((n.mean() - 1) / (self.data.kstar.mean() - 1)) / np.log(r_eff)
            
            # Compute ID error
            id_err = self._compute_binomial_cramerrao(id, self.data.kstar-1, r_eff, self.N)
            
            # Compute likelihood
            log_lik = self._binomial_loglik(id, self.data.kstar - 1, n - 1, r_eff)
            
            # Model validation through KS test
            n_model = self.rng.binomial(self.data.kstar-1, r_eff**id, size=len(n))
            ks, pv = ks_2samp(n-1, n_model)
            
            # Set new ID
            self.data.set_id(id)
            ids[i] = id
            ids_err[i] = id_err
            kstars[i] = self.data.kstar
            log_likelihoods[i] = log_lik
            ks_stats[i] = ks
            p_values[i] = pv

        # Store final results
        self.intrinsic_dim = id
        self.intrinsic_dim_err = id_err
        self.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())
        
        return ids, kstars[(n_iter - 1), :]

    def find_k_nearest_neighbors(self, index, k, cosine=True):
        """
        Find k nearest neighbors for a given embedding index.
        
        Args:
            index: Index of target embedding
            k: Number of neighbors to find
            cosine: If True, use cosine distance; if False, use dot product
            
        Returns:
            list: Indices of k nearest neighbors
        """
        target_embedding = self.embeddings[index]
        
        if cosine:
            # Compute cosine distance
            all_distances = np.array([distance.cosine(target_embedding, emb) 
                                    for emb in self.embeddings])
            # Sort by ascending order (smaller distance = higher similarity)
            nearest_indices = np.argsort(all_distances)[1:k+1]
        else:
            # Compute dot product
            all_scores = util.dot_score(target_embedding, self.embeddings)[0].cpu().tolist()
            # Sort by descending order (larger score = higher similarity)
            nearest_indices = np.argsort(all_scores)[::-1][1:k+1]
            
        return nearest_indices.tolist()

    def _compute_binomial_cramerrao(self, id, k, r, N):
        """Helper method to compute Cramer-Rao bound for binomial estimation."""
        p = r**id
        return np.sqrt(p*(1-p)/(k*N*np.log(r)**2))

    def _binomial_loglik(self, id, k, n, r):
        """Helper method to compute binomial log-likelihood."""
        p = r**id
        return np.sum(n*np.log(p) + (k-n)*np.log(1-p))
