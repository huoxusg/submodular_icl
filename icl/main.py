import os
import argparse
import json
import csv
from glob import glob

from .data_utils import prepare_sft_data, prepare_eval_data, evaluation
# from .model_utils import *


LLM_DIR = os.environ.get("SUBICL_LLM_DIR", "/path/to/huggingface/llm/models/")

datasets = {
    'ner': 'NER_chemprot',
    're': 'RE_DDI',
    'tc': 'TC_healthadvice',
}

example_files = {
    'ner': 'datasets/NER_chemprot_train.json',
    're': 'datasets/RE_DDI_train.json',
    'tc': 'datasets/TC_healthadvice_train.json',
}

query_files = {
    'ner': 'datasets/NER_chemprot_test.json',
    're': 'datasets/RE_DDI_test.json',
    'tc': 'datasets/TC_healthadvice_test.json',
}


def get_arguments():
    parser = argparse.ArgumentParser(description="SFT a model with Hugging Face Transformers")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    # Data
    # parser.add_argument("--example_file", type=str, default="DS_train.json", help="Example data")
    # parser.add_argument("--train_path", type=str, default="DS_train.json", help="Training data")
    # parser.add_argument("--test_path", type=str, default="DS_train.json", help="Test data")
    
    # Model
    parser.add_argument("--llm_name", type=str, default="Qwen", help="Model name")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora_alpha")

    # Training
    # bc2gm: 4, 4, 2000; DDI: 3, 4, 1000; healthadvice: 2, 4, 2000
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # Stats
    parser.add_argument(
        "--stats_mode",
        type=str,
        default="none",
        choices=["none", "bootstrap", "cv_ttest"],
        help="run statistical comparison utilities only",
    )
    parser.add_argument("--file_dir", type=str, default="ragged_datasets", help="directory of results files")
    parser.add_argument("--pattern1", type=str, default="", help="method-1 file prefix")
    parser.add_argument("--pattern2", type=str, default="", help="method-2 file prefix")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="bootstrap resampling count")
    parser.add_argument("--n_folds", type=int, default=10, help="fold count for cv_ttest")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument(
        "--stats_output_dir",
        type=str,
        default="results/stats",
        help="directory for structured stats outputs (json/csv)",
    )
    parser.add_argument(
        "--stats_output_prefix",
        type=str,
        default="",
        help="optional output filename prefix for stats outputs",
    )

    return parser.parse_args()


def sft(args):
    import torch

    torch.manual_seed(args.seed)

    for task in datasets.keys():
        dataset_name = datasets[task]
        
        example_file = os.path.join('datasets/', f'{dataset_name}_train.json')
        query_file = os.path.join('datasets/', f'{dataset_name}_valid.json')
        train_data = prepare_sft_data(example_file)
        test_data = prepare_sft_data(query_file)
        
        if args.llm_name == 'Qwen':
            llm_full_name = 'Qwen2.5-7B'
        elif args.llm_name == 'llama':
            llm_full_name = 'llama-3.1-8B'
        else:
            raise ValueError('Invalid LLM_NAME!')
        llm_name_or_path = os.path.join(LLM_DIR, llm_full_name)
        args.lora_dir = os.path.join('peft_models', f'{dataset_name}_{llm_full_name}_{args.seed}')
        if not os.path.exists(args.lora_dir):
            os.makedirs(args.lora_dir)
        
        model, tokenizer = get_model_and_tokenizer(llm_name_or_path, 'right')
        sft_trainer = get_sft_trainer(train_data, test_data, model, tokenizer, args)
        sft_trainer.train()


def icl(algo, llm_name, seed, per_device_eval_batch_size=64):
    llm_name_or_path = os.path.join(LLM_DIR, llm_name)
    base_model, _ = get_model_and_tokenizer(llm_name_or_path, 'left')
    for task in example_files.keys():
        dataset_name = datasets[task]
        lora_dir = os.path.join('peft_models', f'{dataset_name}_{llm_name}_{seed}')
        model, tokenizer = get_sft_model_and_tokenizer(base_model, lora_dir, 'left')
        
        # for retriever_name in ['bge-large', 'bmretriever', 'medcpt']:
        for retriever_name in ['bmretriever']:
            print(f"Algo {algo} for {task.upper()} using {retriever_name}...")

            example_file = example_files[task]
            query_file = query_files[task]
            test_file = os.path.join('ragged_datasets', f'{algo}_{retriever_name}_' + os.path.basename(query_file))
            eval_file = test_file.replace('test.json', f'test_{llm_name}_{seed}_results.json')
            if not os.path.exists(eval_file):
                prepare_eval_data(example_file,
                                test_file,
                                eval_file,
                                model,
                                tokenizer,
                                per_device_eval_batch_size,
                                )
            evaluation(eval_file)


def icl_permute(pattern, llm_name, seed, per_device_eval_batch_size, retriever):
    llm_name_or_path = os.path.join(LLM_DIR, llm_name)
    base_model, _ = get_model_and_tokenizer(llm_name_or_path, 'left')
    for task in example_files.keys():
        dataset_name = datasets[task]
        lora_dir = os.path.join('peft_models', f'{dataset_name}_{llm_name}_{seed}')
        model, tokenizer = get_sft_model_and_tokenizer(base_model, lora_dir, 'left')
        
        print(f"Pattern {pattern} for {task.upper()} ...")
        example_file = example_files[task]
        query_file = query_files[task]
            
        test_files = glob(os.path.join('ragged_datasets', 'permuted', 
                            f'{pattern}_{retriever}_' + os.path.basename(query_file).replace('.json', '*[0-2].json')))
        print(len(test_files))
        for test_file in test_files:  
            print(test_file) 
            eval_file = test_file.replace('.json', f'_results.json')
            # print(eval_file)
            if not os.path.exists(eval_file):
                prepare_eval_data(example_file,
                                test_file,
                                eval_file,
                                model,
                                tokenizer,
                                per_device_eval_batch_size,
                                )
            evaluation(eval_file)


def evaluate_exisiting_files(file_dir, pattern='*_results.json'):
    from .data_utils import parse_filename

    all_files = glob(os.path.join(file_dir, pattern))
    print(len(all_files), all_files[0])

    for file in all_files:
        res = parse_filename(os.path.basename(file))
        print(file)
        print(f"Evaluating {res['task']} from {res['llm']}, " + \
                f"examples acquired from {res['pattern']} with {res['contriever']}")
        evaluation(file)
        print('\n')


def evaluate_exisiting_files_bootstrp(file_dir, pattern='*_results.json', n_bootstraps=100):
    from .data_utils import parse_filename
    from .metric import get_ner_stats, calculate_ner_metrics, get_classify_stats, calculate_macro_f1, calculate_micro_f1
    import random
    import numpy as np
    from collections import defaultdict

    all_files = glob(os.path.join(file_dir, pattern))
    if not all_files:
        print("No files found.")
        return
    print("[Deprecated] bootstrap on a fixed test set can be biased; prefer compare_methods_cv_ttest.")
    print(len(all_files), all_files[0])

    for file in all_files:
        res = parse_filename(os.path.basename(file))
        print(file)
        print(f"Evaluating {res['task']} from {res['llm']}, " + \
                f"examples acquired from {res['pattern']} with {res['contriever']}")
        
        with open(file, 'r') as fin:
            lines = fin.readlines()
        
        bootstrap_results = defaultdict(list)
        
        for _ in range(n_bootstraps):
            sampled_lines = random.choices(lines, k=len(lines))
            
            if 'NER' in file:
                stats_dict = get_ner_stats(sampled_lines)
                metric = calculate_ner_metrics(stats_dict)
                bootstrap_results['f1'].append(metric['f1'])
                bootstrap_results['precision'].append(metric['precision'])
                bootstrap_results['recall'].append(metric['recall'])
            else:
                acc_stats, prf_stats, _ = get_classify_stats(sampled_lines)
                acc = acc_stats['correct'] / acc_stats['label'] if acc_stats['label'] > 0 else 0
                macro_metric = calculate_macro_f1(prf_stats)
                micro_metric = calculate_micro_f1(prf_stats)
                
                bootstrap_results['accuracy'].append(acc)
                bootstrap_results['macro_f1'].append(macro_metric['macro_f1'])
                bootstrap_results['micro_f1'].append(micro_metric['micro_f1'])
        
        print(f"Bootstrap Results (n={n_bootstraps}):")
        for key, values in bootstrap_results.items():
            mean_score = np.mean(values)
            std_score = np.std(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            print(f"{key}: Mean={mean_score:.4f}, Std={std_score:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        print('\n')


def compare_methods_bootstrap(file_dir, pattern1, pattern2, n_bootstraps=1000):
    from .data_utils import parse_filename
    from .metric import get_ner_stats, calculate_ner_metrics, get_classify_stats, calculate_macro_f1, calculate_micro_f1
    import random
    import numpy as np
    from collections import defaultdict

    # Find files for pattern1
    files1 = glob(os.path.join(file_dir, f'{pattern1}*_results.json'))
    if not files1:
        print(f"No files found for pattern: {pattern1}")
        return
    
    print(f"Comparing {pattern1} vs {pattern2} (n_bootstraps={n_bootstraps})")
    print("[Deprecated] bootstrap on a fixed test set can be biased; prefer compare_methods_cv_ttest.")
    print(f"Found {len(files1)} files for {pattern1}")

    for file1 in files1:
        basename1 = os.path.basename(file1)
        if not basename1.startswith(pattern1):
            continue
            
        # Construct corresponding file2 path
        basename2 = basename1.replace(pattern1, pattern2, 1)
        file2 = os.path.join(file_dir, basename2)
        
        if not os.path.exists(file2):
            print(f"Skipping: Matching file for {basename1} not found: {basename2}")
            continue
            
        res = parse_filename(basename1)
        print(f"Comparing Task: {res['task']}, LLM: {res['llm']}, Retriever: {res['contriever']}")
        print(f"File 1: {basename1}")
        print(f"File 2: {basename2}")

        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            
        if len(lines1) != len(lines2):
            print(f"Warning: Line counts differ ({len(lines1)} vs {len(lines2)}). Skipping.")
            continue
            
        n_samples = len(lines1)
        diff_results = defaultdict(list)
        
        for _ in range(n_bootstraps):
            # Sample indices with replacement
            indices = random.choices(range(n_samples), k=n_samples)
            
            sample1 = [lines1[i] for i in indices]
            sample2 = [lines2[i] for i in indices]
            
            if 'NER' in basename1:
                stats1 = get_ner_stats(sample1)
                metric1 = calculate_ner_metrics(stats1)
                
                stats2 = get_ner_stats(sample2)
                metric2 = calculate_ner_metrics(stats2)
                
                diff_results['f1_diff'].append(metric1['f1'] - metric2['f1'])
                diff_results['precision_diff'].append(metric1['precision'] - metric2['precision'])
                diff_results['recall_diff'].append(metric1['recall'] - metric2['recall'])
            else:
                acc_stats1, prf_stats1, _ = get_classify_stats(sample1)
                acc1 = acc_stats1['correct'] / acc_stats1['label'] if acc_stats1['label'] > 0 else 0
                macro1 = calculate_macro_f1(prf_stats1)['macro_f1']
                
                acc_stats2, prf_stats2, _ = get_classify_stats(sample2)
                acc2 = acc_stats2['correct'] / acc_stats2['label'] if acc_stats2['label'] > 0 else 0
                macro2 = calculate_macro_f1(prf_stats2)['macro_f1']
                
                diff_results['accuracy_diff'].append(acc1 - acc2)
                diff_results['macro_f1_diff'].append(macro1 - macro2)

        print(f"Bootstrap Difference Results ({pattern1} - {pattern2}):")
        for key, values in diff_results.items():
            mean_diff = np.mean(values)
            std_diff = np.std(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            significant = "SIGNIFICANT" if (ci_lower > 0 or ci_upper < 0) else "NOT SIGNIFICANT"
            print(f"{key}: Mean Diff={mean_diff:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}] ({significant})")
        print('-' * 50 + '\n')


def _compute_metrics_from_lines(lines, is_ner):
    from .metric import get_ner_stats, calculate_ner_metrics, get_classify_stats, calculate_macro_f1, calculate_micro_f1

    if is_ner:
        stats_dict = get_ner_stats(lines)
        metric = calculate_ner_metrics(stats_dict)
        return {
            'f1': metric['f1'],
            'precision': metric['precision'],
            'recall': metric['recall'],
        }

    acc_stats, prf_stats, _ = get_classify_stats(lines)
    acc = acc_stats['correct'] / acc_stats['label'] if acc_stats['label'] > 0 else 0
    macro_metric = calculate_macro_f1(prf_stats)
    micro_metric = calculate_micro_f1(prf_stats)
    return {
        'accuracy': round(acc, 4),
        'macro_f1': macro_metric['macro_f1'],
        'micro_f1': micro_metric['micro_f1'],
    }


def _build_kfold_indices(n_samples, n_folds, seed):
    import numpy as np

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_samples < n_folds:
        raise ValueError(f"n_samples({n_samples}) < n_folds({n_folds}); reduce n_folds")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    return [fold.tolist() for fold in np.array_split(shuffled, n_folds)]


def _paired_ttest_from_scores(scores1, scores2, alpha):
    import numpy as np

    arr1 = np.asarray(scores1, dtype=float)
    arr2 = np.asarray(scores2, dtype=float)
    diffs = arr1 - arr2
    n = len(diffs)
    if n < 2:
        raise ValueError("paired t-test requires at least 2 folds")

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required for paired t-test. Please install scipy.") from e

    t_stat, p_value = stats.ttest_rel(arr1, arr2)

    if std_diff == 0.0:
        ci_lower = mean_diff
        ci_upper = mean_diff
    else:
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2.0, df=n - 1)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

    significant = "SIGNIFICANT" if p_value < alpha else "NOT SIGNIFICANT"
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'significant': significant,
    }


def compare_methods_cv_ttest(
    file_dir,
    pattern1,
    pattern2,
    n_folds=10,
    seed=42,
    alpha=0.05,
    output_json_path=None,
    output_csv_path=None,
):
    from .data_utils import parse_filename
    from collections import defaultdict

    files1 = glob(os.path.join(file_dir, f'{pattern1}*_results.json'))
    if not files1:
        print(f"No files found for pattern: {pattern1}")
        return

    print(f"Comparing {pattern1} vs {pattern2} (n_folds={n_folds}, seed={seed}, alpha={alpha})")
    print(f"Found {len(files1)} files for {pattern1}")
    print("Method note: posterior fold-splitting on fixed predictions (not full retrain/reinference CV).")
    all_records = []

    for file1 in files1:
        basename1 = os.path.basename(file1)
        if not basename1.startswith(pattern1):
            continue

        basename2 = basename1.replace(pattern1, pattern2, 1)
        file2 = os.path.join(file_dir, basename2)
        if not os.path.exists(file2):
            print(f"Skipping: Matching file for {basename1} not found: {basename2}")
            continue

        res = parse_filename(basename1)
        print(f"Comparing Task: {res['task']}, LLM: {res['llm']}, Retriever: {res['contriever']}")
        print(f"File 1: {basename1}")
        print(f"File 2: {basename2}")

        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            print(f"Warning: Line counts differ ({len(lines1)} vs {len(lines2)}). Skipping.")
            continue

        n_samples = len(lines1)
        if n_samples < n_folds:
            print(f"Warning: samples={n_samples} < n_folds={n_folds}. Skipping.")
            continue

        fold_indices = _build_kfold_indices(n_samples, n_folds, seed)
        is_ner = 'NER' in basename1

        fold_scores1 = defaultdict(list)
        fold_scores2 = defaultdict(list)

        for fold in fold_indices:
            fold_lines1 = [lines1[i] for i in fold]
            fold_lines2 = [lines2[i] for i in fold]

            metrics1 = _compute_metrics_from_lines(fold_lines1, is_ner)
            metrics2 = _compute_metrics_from_lines(fold_lines2, is_ner)

            for key in metrics1:
                fold_scores1[key].append(metrics1[key])
                fold_scores2[key].append(metrics2[key])

        print(f"Paired t-test Results ({pattern1} - {pattern2}):")
        for key in fold_scores1:
            stats_res = _paired_ttest_from_scores(fold_scores1[key], fold_scores2[key], alpha)
            diff_scores = [a - b for a, b in zip(fold_scores1[key], fold_scores2[key])]
            record = {
                'pattern1': pattern1,
                'pattern2': pattern2,
                'task': res['task'],
                'llm': res['llm'],
                'retriever': res['contriever'],
                'metric': key,
                'n_folds': n_folds,
                'seed': seed,
                'alpha': alpha,
                'file1': basename1,
                'file2': basename2,
                'fold_scores1': [float(v) for v in fold_scores1[key]],
                'fold_scores2': [float(v) for v in fold_scores2[key]],
                'fold_diffs': [float(v) for v in diff_scores],
                'mean_diff': stats_res['mean_diff'],
                'std_diff': stats_res['std_diff'],
                't_stat': stats_res['t_stat'],
                'p_value': stats_res['p_value'],
                'ci_lower': stats_res['ci_lower'],
                'ci_upper': stats_res['ci_upper'],
                'significant': stats_res['significant'],
            }
            all_records.append(record)
            print(
                f"{key}_diff: Mean Diff={stats_res['mean_diff']:.4f}, "
                f"Std Diff={stats_res['std_diff']:.4f}, "
                f"t={stats_res['t_stat']:.4f}, p={stats_res['p_value']:.6f}, "
                f"95% CI=[{stats_res['ci_lower']:.4f}, {stats_res['ci_upper']:.4f}] "
                f"({stats_res['significant']})"
            )
        print('-' * 50 + '\n')

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as fout:
            json.dump(all_records, fout, ensure_ascii=False, indent=2)
        print(f"Saved JSON stats to: {output_json_path}")

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        fieldnames = [
            'pattern1',
            'pattern2',
            'task',
            'llm',
            'retriever',
            'metric',
            'n_folds',
            'seed',
            'alpha',
            'file1',
            'file2',
            'mean_diff',
            'std_diff',
            't_stat',
            'p_value',
            'ci_lower',
            'ci_upper',
            'significant',
            'fold_scores1',
            'fold_scores2',
            'fold_diffs',
        ]
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_records:
                csv_row = dict(row)
                csv_row['fold_scores1'] = ';'.join(f"{v:.6f}" for v in row['fold_scores1'])
                csv_row['fold_scores2'] = ';'.join(f"{v:.6f}" for v in row['fold_scores2'])
                csv_row['fold_diffs'] = ';'.join(f"{v:.6f}" for v in row['fold_diffs'])
                writer.writerow(csv_row)
        print(f"Saved CSV stats to: {output_csv_path}")

    return all_records


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = get_arguments()

    if args.stats_mode != 'none':
        if not args.pattern1 or not args.pattern2:
            raise ValueError("--pattern1 and --pattern2 are required when --stats_mode is enabled")

        if args.stats_mode == 'bootstrap':
            compare_methods_bootstrap(args.file_dir, args.pattern1, args.pattern2, n_bootstraps=args.n_bootstraps)
        elif args.stats_mode == 'cv_ttest':
            if os.path.exists(args.stats_output_dir) and not os.path.isdir(args.stats_output_dir):
                raise ValueError(
                    f"--stats_output_dir points to an existing file: {args.stats_output_dir}. "
                    "Please provide a directory path."
                )
            prefix = args.stats_output_prefix.strip()
            if not prefix:
                prefix = f"cv_ttest_{args.pattern1}_vs_{args.pattern2}_{args.n_folds}fold_seed{args.seed}"
            output_json_path = os.path.join(args.stats_output_dir, f"{prefix}.json")
            output_csv_path = os.path.join(args.stats_output_dir, f"{prefix}.csv")
            compare_methods_cv_ttest(
                args.file_dir,
                args.pattern1,
                args.pattern2,
                n_folds=args.n_folds,
                seed=args.seed,
                alpha=args.alpha,
                output_json_path=output_json_path,
                output_csv_path=output_csv_path,
            )
        raise SystemExit(0)
    
    # sft(args)

    
    if args.llm_name == 'Qwen':
        pass
        # icl('most_similar', 'Qwen2.5-7B', args.seed, 40)
        # icl('random_similar_100_3', 'Qwen2.5-7B', args.seed, 40)
        # icl('random_similar_100_5', 'Qwen2.5-7B', args.seed, 40)
        # icl('s3_single', 'Qwen2.5-7B', args.seed, 40)
        # icl('s3_whole', 'Qwen2.5-7B', args.seed, 40)
        # icl('reg_s3_100_30_3', 'Qwen2.5-7B', args.seed, 40)
        # icl('s3_reg_100_30_3', 'Qwen2.5-7B', args.seed, 40)
        # icl('dual_reg_100_30_3', 'Qwen2.5-7B', args.seed, 40)

        # icl_permute('random_similar_100_3', 'Qwen2.5-7B', args.seed, 40, 'bge-large')

        # icl_permute('dual_reg_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'medcpt')
        # icl_permute('s3_reg_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'medcpt')
        # icl_permute('reg_s3_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'medcpt')
        # icl_permute('s3_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'medcpt')
        # icl_permute('random_similar_100_3', 'Qwen2.5-7B', args.seed, 40, 'medcpt')

        # icl_permute('dual_reg_100_30_3', 'Qwen2.5-7B', args.seed, 48, 'bmretriever')
        # icl_permute('s3_reg_100_30_3', 'Qwen2.5-7B', args.seed, 48, 'bmretriever')
        # icl_permute('reg_s3_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'bmretriever')
        # icl_permute('s3_100_30_3', 'Qwen2.5-7B', args.seed, 40, 'bmretriever')
        # icl_permute('random_similar_100_3', 'Qwen2.5-7B', args.seed, 40, 'bmretriever')
    
    elif args.llm_name == 'llama':
        pass
        # icl('most_similar', 'llama-3.1-8B', args.seed, 40)
        # icl('random_similar_100_3', 'llama-3.1-8B', args.seed, 40)
        # icl('random_similar_100_5', 'llama-3.1-8B', args.seed, 40)
        # icl('s3_single', 'llama-3.1-8B', args.seed, 40)
        # icl('s3_100_30_3', 'llama-3.1-8B', args.seed, 40)
        # icl('reg_s3_100_30_3', 'llama-3.1-8B', args.seed, 40)
        # icl('s3_reg_100_30_3', 'llama-3.1-8B', args.seed, 40)
        # icl('dual_reg_100_30_3', 'llama-3.1-8B', args.seed, 40)
    else:
        raise ValueError('Invalid LLM!')

    # evaluate_exisiting_files('ragged_datasets', 'random_similar_100_3*_results.json')
    # evaluate_exisiting_files('ragged_datasets', '*_3_*bmretriever*Qwen*_results.json')


    # 评估单个文件或匹配模式的文件，并进行 bootstrap 检验
    # evaluate_exisiting_files_bootstrp('ragged_datasets', 'random_similar_100_3*_results.json', n_bootstraps=1000)

    # 比较两个方法的 bootstrap 差异
    # compare_methods_bootstrap('ragged_datasets', 'random_similar_100_3', 'dual_reg_100_30_3', n_bootstraps=1000)

    # 10-fold paired t-test（后验折分）
    compare_methods_cv_ttest('ragged_datasets', 'random_similar_100_3', 'dual_reg_100_30_3', n_folds=10, seed=42)
