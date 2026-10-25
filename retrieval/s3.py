import sys
import time
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch


class S3:
    def __init__(self, query_npy, example_npy, example_indices, query_idx=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.query_matrix = torch.tensor(np.load(query_npy), dtype=torch.float32, device=self.device)
        self.example_matrix = torch.tensor(np.load(example_npy), dtype=torch.float32, device=self.device)
        # print(f"Example matrix shape: {self.example_matrix.shape}")
        # print(f"Query matrix shape: {self.query_matrix.shape}")
        self.query_matrix = self.normalize(self.query_matrix)
        self.example_matrix = self.normalize(self.example_matrix)
        self.recall_indices = example_indices
        if query_idx is not None: # for query-single algos
            self.query_matrix = self.query_matrix[query_idx: (query_idx+1)]
    
    @staticmethod
    def normalize(matrix):
        norms = torch.norm(matrix, dim=1, keepdim=True)
        return matrix / norms
    
    def calc_coverage_gain(self, vec, penalize, lmbda):
        '''Given an embedding vector from example matrix, calculate f(a|Q)
        '''
        vec = vec.reshape(1, -1).to(self.device)
        assert vec.shape[1] == self.query_matrix.shape[1]
        query_matrix = torch.cat((self.query_matrix, vec), dim=0)
        dot_matrix = (torch.mm(self.example_matrix, query_matrix.T) + 1.) / 2.
        score = torch.sum(torch.max(dot_matrix, dim=1).values).item()
        if penalize:
            div_score = self.calc_div_score(query_matrix)
            # print(score, lmbda * div_score)
            score += lmbda * div_score
        return score

    def run(self, k1, penalize=False, lmbda=0.):
        example_gains = []
        for ii in tqdm(self.recall_indices):
            example_vec = self.example_matrix[ii]
            gain = self.calc_coverage_gain(example_vec, penalize, lmbda)
            example_gains.append(gain)
        top_gains, top_indices = torch.topk(torch.tensor(example_gains, device=self.device), k1)
        # print(top_indices)
        origin_indices = [self.recall_indices[idx] for idx in top_indices.cpu().numpy()]
        return origin_indices
        # return self.example_matrix.cpu().numpy()[origin_indices]
    
    def calc_div_score(self, matrix, eps=1e-6):
        div_matrix = torch.mm(matrix, matrix.T)
        ## numeric stability
        div_matrix = (div_matrix + div_matrix.T) / 2.
        div_matrix += eps * torch.eye(div_matrix.shape[0], device=self.device)

        # eigenvalues = torch.linalg.eigvalsh(div_matrix)  # for symmetric matrices
        # clipped_eigenvalues = torch.clamp(eigenvalues, min=eps)
        # div_score = torch.sum(torch.log(clipped_eigenvalues))

        eigenvalues = np.linalg.eigvalsh(div_matrix.cpu().numpy())  # for symmetric matrices
        clipped_eigenvalues = np.maximum(eigenvalues, eps)
        div_score = np.sum(np.log(clipped_eigenvalues))
        return div_score.item()
