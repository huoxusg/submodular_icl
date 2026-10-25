import os
import json
from .algorithm import GreedyRecall
from .s3 import S3
from .constants import *


def prepare_embeddings():
    from icl.model_utils import get_retrieve_model
    from .utils import generate_sent_embedding

    # for example_file in query_files.values():
    for example_file in list(query_files.values()) + list(example_files.values()):
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
        # for contrieve_name in ['bge-large']:
            print(f"Generating embeddings for {example_file} using {contrieve_name}...")
            model, tokenizer = get_retrieve_model(retriever_mappings[contrieve_name], cache_dir)
            # print(model, tokenizer)
            # print(tokenizer.cls_token_id, tokenizer.pad_token_id)

            output_file = example_file.replace('.json', f'_embeddings_{contrieve_name}.npy').replace('datasets', 'npy_files')
            generate_sent_embedding(example_file,
                                    output_file,
                                    model,
                                    tokenizer,
                                    pooling_mappings[contrieve_name],
                                    batch_size=32,)
            # print()


def summarize_example_files(num=100):
    import numpy as np
    from .algorithm import GreedyRecall

    # npy_file = 'datasets/NER_bc2gm_train_embeddings_contriever.npy'
    for example_file in example_files.values():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            print(f"Greedy for {example_file} using {contrieve_name}...")
            npy_file = example_file.replace('.json', f'_embeddings_{contrieve_name}.npy')
            
            algo = GreedyRecall(npy_file)
            res = algo.greedy_loop(num)
            print(res)
            print(algo.calc_div_score(res))
            
            res = algo.greedy_loop(num, True)
            print(res)
            print(algo.calc_div_score(res))
            
            break
        break


def finalize_examples_s3_whole(recall_num, k1, k2):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"S3 with whole query set for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            ## Recall from example files
            algo = GreedyRecall(example_npy_file)
            indices = algo.greedy_loop(recall_num)

            ## Phase 1 of S3
            algo = S3(query_npy_file, example_npy_file, indices)
            indices = algo.run(k1)
            print(indices)

            ## Phase 2 of S3
            algo = GreedyRecall(example_npy_file, partial_indices=indices)
            tmp_indices = algo.greedy_loop(k2)
            indices = [int(indices[ii]) for ii in tmp_indices]
            print(indices)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f's3_{recall_num}_{k1}_{k2}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for line in fin.readlines():
                        query = json.loads(line.strip())
                        query['example_indices'] = indices
                        print(json.dumps(query), file=fout)
            
        # break

def finalize_examples_reg_s3_whole(recall_num, k1, k2):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"\nDiv (Reg) - S3 with whole query set for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            ## Recall from example files
            coef = task_contriever_coef_mappings[task][contrieve_name]
            algo = GreedyRecall(example_npy_file)
            indices = algo.greedy_loop(recall_num, True, coef)

            ## Phase 1 of S3
            algo = S3(query_npy_file, example_npy_file, indices)
            indices = algo.run(k1)
            print(indices)

            ## Phase 2 of S3
            algo = GreedyRecall(example_npy_file, partial_indices=indices)
            tmp_indices = algo.greedy_loop(k2)
            indices = [int(indices[ii]) for ii in tmp_indices]
            print(indices)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f'reg_s3_{recall_num}_{k1}_{k2}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for line in fin.readlines():
                        query = json.loads(line.strip())
                        query['example_indices'] = indices
                        print(json.dumps(query), file=fout)


def finalize_examples_s3_reg_whole(recall_num, k1, k2):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"\nDiv - S3 (Reg) with whole query set for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            ## Recall from example files
            algo = GreedyRecall(example_npy_file)
            indices = algo.greedy_loop(recall_num)

            ## Phase 1 of S3
            coef = task_contriever_coef_mappings[task][contrieve_name]
            algo = S3(query_npy_file, example_npy_file, indices)
            indices = algo.run(k1, True, coef)
            print(indices)

            ## Phase 2 of S3
            algo = GreedyRecall(example_npy_file, partial_indices=indices)
            tmp_indices = algo.greedy_loop(k2)
            indices = [int(indices[ii]) for ii in tmp_indices]
            print(indices)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f's3_reg_{recall_num}_{k1}_{k2}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for line in fin.readlines():
                        query = json.loads(line.strip())
                        query['example_indices'] = indices
                        print(json.dumps(query), file=fout)


def finalize_examples_dual_reg_whole(recall_num, k1, k2):
    for task in example_files.keys():
        for contrieve_name in ['bge-large', 'bmretriever', 'medcpt']:
            example_file = example_files[task]
            query_file = query_files[task]
            print(f"\nDual Reg with whole query set for {task.upper()} using {contrieve_name}...")
            example_npy_file = os.path.join(npy_dir, os.path.basename(example_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))
            query_npy_file = os.path.join(npy_dir, os.path.basename(query_file).replace('.json', f'_embeddings_{contrieve_name}.npy'))

            coef = task_contriever_coef_mappings[task][contrieve_name]

            ## Recall from example files
            algo = GreedyRecall(example_npy_file)
            indices = algo.greedy_loop(recall_num, True, coef)

            ## Phase 1 of S3
            
            algo = S3(query_npy_file, example_npy_file, indices)
            indices = algo.run(k1, True, coef)
            print(indices)

            ## Phase 2 of S3
            algo = GreedyRecall(example_npy_file, partial_indices=indices)
            tmp_indices = algo.greedy_loop(k2)
            indices = [int(indices[ii]) for ii in tmp_indices]
            print(indices)

            ## output and save to file
            output_file = os.path.join(ragged_dir, f'dual_reg_{recall_num}_{k1}_{k2}_{contrieve_name}_' + os.path.basename(query_file))
            with open(query_file, 'r') as fin:
                with open(output_file, 'w') as fout:
                    for line in fin.readlines():
                        query = json.loads(line.strip())
                        query['example_indices'] = indices
                        print(json.dumps(query), file=fout)


def rename_file(file_dir):
    from glob import glob

    all_files = glob(os.path.join(file_dir, '*.json'))
    print(len(all_files), all_files[0])

    num = 0
    for old_name in all_files:
        if 'random_similar' in old_name:
            if '_100_' not in old_name:
                new_name = old_name.replace('similar_', 'similar_100_3_')
                os.rename(old_name, new_name)
        elif 'single' in old_name:
            continue
        else:
            assert '_whole_' in old_name
            num += 1
            if '_100_30' in old_name:
                new_name = old_name.replace('_whole_', '_')
                os.rename(old_name, new_name)
            else:
                new_name = old_name.replace('_whole_', '_100_30_3_')
                os.rename(old_name, new_name)
    print(num)


def permute_file(input_dir, pattern, output_dir):
    from glob import glob
    orders = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

    all_files = glob(os.path.join(input_dir, pattern))
    print(len(all_files), all_files[0])

    for file in all_files:
        print(file)
        for order in orders:
            suffix = f"_{order[0]}_{order[1]}_{order[2]}"
            output_file = os.path.join(output_dir, os.path.basename(file).replace('.json', f'{suffix}.json'))
            # print(output_file)
            with open(output_file, 'w') as fout:
                with open(file, 'r') as fin:
                    for line in fin.readlines():
                        query = json.loads(line.strip())
                        # print(query.keys())
                        # print(query['example_indices'])
                        example_indices = [int(ii) for ii in query['example_indices']]
                        assert len(example_indices) == 3
                        new_indices = [example_indices[ii] for ii in order]
                        query['example_indices'] = new_indices
                        print(json.dumps(query), file=fout)


if __name__ == '__main__':
    from .similar_based import finalize_examples_most_similar, finalize_examples_random_similar
    
    pass
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # prepare_embeddings()
    # summarize_example_files()

    # finalize_examples_most_similar(3)
    # finalize_examples_random_similar(100, 3)

    # finalize_examples_s3_whole(100, 30, 3)

    # finalize_examples_reg_s3_whole(100, 30, 3)

    # finalize_examples_s3_reg_whole(100, 30, 3)

    # finalize_examples_dual_reg_whole(100, 30, 3)


    # finalize_examples_random_similar(100, 5)
    # finalize_examples_dual_reg_whole(100, 30, 5)

    ## rename_file('ragged_datasets')
    # permute_file('ragged_datasets', '*_3_bge-large*test.json', 'ragged_datasets/permuted')