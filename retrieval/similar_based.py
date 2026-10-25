import os, sys
import time
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch


class SimilarRank:
    def __init__(self, query_npy, example_npy):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.query_matrix = torch.tensor(np.load(query_npy), dtype=torch.float32, device=self.device)
        self.example_matrix = torch.tensor(np.load(example_npy), dtype=torch.float32, device=self.device)
        print(f"Example matrix shape: {self.example_matrix.shape}")
        print(f"Query matrix shape: {self.query_matrix.shape}")
        self.query_matrix = self.normalize(self.query_matrix)
        self.example_matrix = self.normalize(self.example_matrix)
    
    @staticmethod
    def normalize(matrix):
        norms = torch.norm(matrix, dim=1, keepdim=True)
        return matrix / norms
    
    def run(self, k):
        dot_matrix = (torch.mm(self.query_matrix, self.example_matrix.T) + 1.) / 2.
        _, indices = torch.topk(dot_matrix, k=k, dim=1)
        return indices.cpu().numpy()
    
    def run_with_random_recall(self, k1, k2, seed):
        torch.manual_seed(seed)
        
        # Randomly select k1 indices
        total_indices = self.example_matrix.shape[0]
        random_indices = torch.randperm(total_indices)[: k1].to(self.device)
        
        # Subset the example matrix
        subset_example_matrix = self.example_matrix[random_indices]
        
        # Compute dot product with the subset
        dot_matrix = (torch.mm(self.query_matrix, subset_example_matrix.T) + 1.) / 2.
        sim_res = torch.sum(dot_matrix, dim=0)
        return torch.topk(sim_res, k=k2, dim=0)[1].cpu().numpy()
        
        # _, subset_indices = torch.topk(dot_matrix, k=k2, dim=1)
        # # Recover original indices
        # original_indices = random_indices[subset_indices].cpu().numpy()
        # return original_indices



from .constants import *
import json


def finalize_examples_most_similar(k=3):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"Most similar for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            algo = SimilarRank(query_npy_file, example_npy_file)
            indices = algo.run(k)
            print(indices.shape)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f'most_similar_{k}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for ii, line in enumerate(fin.readlines()):
                        query = json.loads(line.strip())
                        query['example_indices'] = indices[ii].tolist()
                        print(json.dumps(query), file=fout)


def finalize_examples_random_similar(k1=100, k2=3, seed=28):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"Random similar for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            algo = SimilarRank(query_npy_file, example_npy_file)
            indices = algo.run_with_random_recall(k1, k2, seed)
            print(indices)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f'random_similar_{k1}_{k2}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for ii, line in enumerate(fin.readlines()):
                        query = json.loads(line.strip())
                        query['example_indices'] = indices.tolist()
                        print(json.dumps(query), file=fout)
