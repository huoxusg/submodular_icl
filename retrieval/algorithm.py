import sys
import time
from copy import deepcopy
import numpy as np
# from tqdm import tqdm

from .queue import UtilityPair, PriorityQueue


class GreedyRecall:
    def __init__(self, npy_file, partial_indices=None, print_mode=True):
        self.matrix = np.load(npy_file)
        if partial_indices is not None:
            self.matrix = self.matrix[partial_indices]
        if print_mode:
            print(f"Matrix shape: {self.matrix.shape}")
        self.print_mode = print_mode
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / norms
        self.num = self.matrix.shape[0]
    
    @staticmethod
    def _initial_coverage_score(matrix, penalize, lmbda):
        dot_matrix = (np.dot(matrix, matrix.T) + 1.) / 2.
        cov_scores = np.sum(dot_matrix, axis=1)
        if penalize:
            cov_scores += lmbda * 1.
            # cov_scores += 0.
        return cov_scores
    
    def initialize_queue(self, penalize, lmbda):
        cos_sim_res = self._initial_coverage_score(self.matrix, penalize, lmbda)
        assert cos_sim_res.shape[0] == self.num
        utility_queue = PriorityQueue()
        ## add one-element marginal utility gain for each idx
        for idx, cos_sim in enumerate(cos_sim_res):
            utility_queue.put(UtilityPair(idx, cos_sim))
        self.utility_queue = utility_queue
    
    def _coverage_score(self, indices, penalize, lmbda, verbose):
        assert type(indices) == set
        indices = np.sort(list(indices))
        dot_matrix = (np.dot(self.matrix, self.matrix[indices].T) + 1.) / 2.
        utility_score = np.sum(np.max(dot_matrix, axis=1))
        if penalize:
            ## v1
            # div_matrix = dot_matrix[indices]
            # div_score = np.sum(div_matrix)
            # utility_score -= lmbda * div_score 
            ## v2
            div_score = self.calc_div_score(indices)
            if verbose:
                print(len(indices), div_score, utility_score, utility_score + lmbda * div_score)
            utility_score += lmbda * div_score
            
        return utility_score

    def greedy_loop(self, budget, penalize=False, lmbda=1., verbose=False):
        if self.print_mode:
            print('Initializing priority queue...')
        st = time.time()
        self.initialize_queue(penalize, lmbda)
        et = time.time()
        
        opt_indices = set()
        budget_left = budget
        if self.print_mode:
            print('Starting greedy loop...')
        while not self.utility_queue.empty():
            if len(opt_indices) >= budget:
                break
            
            pair = self.utility_queue.get()
            ## lazy greedy
            if opt_indices:
                cur_val = self._coverage_score(opt_indices, penalize, lmbda, verbose)
                tmp_indices = deepcopy(opt_indices)
                tmp_indices.add(pair.index)
                new_val = self._coverage_score(tmp_indices, penalize, lmbda, verbose)
                gain = new_val - cur_val
            else:
                gain = pair.value

            # if gain <= 0:
            #     break
            if gain >= self.utility_queue.top_priority():
                budget_left -= 1.
                opt_indices.add(pair.index)
            else:
                self.utility_queue.put(UtilityPair(pair.index, gain))
        
        if self.print_mode:
            print(f'Greedy algorithm finished. Spending {round(et - st, 1)}s initializing and {round(time.time() - et, 1)}s in loop.')
        # sys.stdout.flush()
        return np.sort(list(opt_indices))

    def calc_div_score(self, indices, eps=1e-6):
        div_matrix = np.dot(self.matrix[indices], self.matrix[indices].T)
        ## numeric stability
        div_matrix = (div_matrix + div_matrix.T) / 2.
        div_matrix += eps * np.eye(div_matrix.shape[0])
        
        # div_score = np.log(np.linalg.det(div_matrix)) # numeric problems, -inf
        
        # L = np.linalg.cholesky(div_matrix)
        # div_score = 2 * np.sum(np.log(np.diag(L)))

        eigenvalues = np.linalg.eigvalsh(div_matrix)  # for symmetric matrices
        clipped_eigenvalues = np.maximum(eigenvalues, eps)
        div_score = np.sum(np.log(clipped_eigenvalues))

        return div_score

    def calc_div_metrics(self, indices):
        div_matrix = np.dot(self.matrix[indices], self.matrix[indices].T)
        div_matrix = (div_matrix + div_matrix.T) / 2.
        
        ilad = np.sum(1. - div_matrix) / (div_matrix.shape[0] ** 2 - div_matrix.shape[0])
        ilmd = np.min(1. - div_matrix + 10 * np.eye(div_matrix.shape[0]))
        return round(ilad.item(), 3), round(ilmd.item(), 3)