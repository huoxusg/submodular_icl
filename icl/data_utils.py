import json


def get_test_prompt(query, include_examples):
    # assert type(query) == dict
    if include_examples:
        assert 'example_text' in query.keys() and query['example_text'] != ''
        input_prompt = '### Instruction:\n' + \
                    f"{query['instruction']}\n\n" + \
                    "### Examples:\n" + \
                    f"{query['example_text']}\n\n" + \
                    f"### Input: \n" + \
                    f"{query['inputsentence']}\n\n" + \
                    f"### Response: \n"
    else:
        input_prompt = ("### Instruction:\n"
                    f"{query['instruction']}\n\n"
                    f"### Input: \n"
                    f"{query['inputsentence']}\n\n"
                    f"### Response: \n")
    return input_prompt


def get_train_prompt(query):
    return get_test_prompt(query, include_examples=False) + f"{query['response']}"


def make_inference(model, tokenizer, query):
    import torch

    prompt = get_test_prompt(query, include_examples=True)
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=120)
    res = (tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(res)
    # print('---- NON-INSTRUCT-TUNED-MODEL ----')
    return res


def process_rag(example_file, query_file):
    ## load files
    with open(example_file, 'r') as fin:
        all_examples = [json.loads(line.strip()) for line in fin.readlines()]
    with open(query_file, 'r') as fin:
        all_data = [json.loads(line.strip()) for line in fin.readlines()]
    assert type(all_data) == list
    print(len(all_data))

    ## rag process
    for data in all_data:
        indices = data['example_indices']
        del data['example_indices']
        rag_examples = [all_examples[i] for i in indices]
        data['example_text'] = ''
        for example in rag_examples:
            data['example_text'] += '#Input:' + example['inputsentence'] + '\n#Response:' + example['response'] + '\n'
    
    return all_data


def prepare_sft_data(example_file):
    from datasets import Dataset
    
    ## load files
    with open(example_file, 'r') as fin:
        all_data = [json.loads(line.strip()) for line in fin.readlines()]

    data_dict = {key: [data[key] for data in all_data] for key in all_data[0].keys()}
    print(data_dict.keys())
    dataset = Dataset.from_dict(data_dict)
    print(dataset)
    sft_data = dataset.map(lambda x: {'text': get_train_prompt(x)})
    # sft_data = dataset.map(lambda x: {'text': get_train_prompt(x), 'labels': x['response']})
    # print(sft_data)
    return sft_data


def prepare_eval_data(example_file,
                      query_file,
                      save_file,
                      model,
                      tokenizer,
                      batch_size=4,
                      monitor_interval=1000,
    ):
    # save_file = args.gen_dir + '/' + 'test_generation.json'
    # save_file = os.path.join(args.lora_path, f'best_model_{args.seed}', 'test_generation.json')
    
    ragged_data = process_rag(example_file, query_file)

    ## make inference for batch data
    prompts = [get_test_prompt(data, include_examples=True) for data in ragged_data]
    responses = []
    start_idx = 0
    max_token_length = 0
    while start_idx < len(prompts):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch = prompts[start_idx: end_idx]
        inputs = tokenizer(batch, return_tensors='pt', return_token_type_ids=False, padding=True, truncation=True).to(model.device)
        model.eval()
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=120)
        responses += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if end_idx // monitor_interval != start_idx // monitor_interval:
            max_token_length = max(inputs['input_ids'].shape[1], max_token_length)
            print(start_idx, end_idx, max_token_length)
            max_token_length = 0
        start_idx = end_idx
    assert len(responses) == len(ragged_data)

    with open(save_file, 'w') as fout:
        for data, response in zip(ragged_data, responses):
            # prediction = make_inference(model, tokenizer, data)
            # print(prediction)
            res = dict()
            res['input'] = data['inputsentence']
            res['demonstration'] = data['example_text']
            res['label'] = data['response']
            res['prediction'] = response
            print(json.dumps(res), file=fout)


def evaluation(eval_file):
    # from .metric import evaluation_ner_v1, evaluation_ner_v2, evaluation_ner_v3
    from .metric import evaluation_classify, evaluation_ner_v4

    if 'NER' in eval_file:
        # evaluation_ner_v2(eval_file)
        # return evaluation_ner_v3(eval_file)
        return evaluation_ner_v4(eval_file)
    else:
        return evaluation_classify(eval_file)


def parse_filename(filename):
    parts = filename.split('_')
    contrievers = {'bge-large', 'bmretriever', 'medcpt'}
    
    # Find contriever's position
    contriever_index = None
    for i, part in enumerate(parts):
        if part in contrievers:
            contriever_index = i
            break
    if contriever_index is None:
        raise ValueError("Contriever not found in the filename.")
    
    # Locate 'test' after contriever
    remaining_after_contriever = parts[contriever_index + 1:]
    try:
        test_rel_index = remaining_after_contriever.index('test')
    except ValueError:
        raise ValueError("'test' not found after contriever in the filename.")
    test_index = contriever_index + 1 + test_rel_index
    
    # Extract task and validate
    task_parts = parts[contriever_index + 1: test_index]
    task = '_'.join(task_parts)
    valid_task_prefixes = ('NER', 'TC', 'RE')
    if not task.startswith(valid_task_prefixes):
        raise ValueError(f"Task '{task}' does not start with NER, TC, or RE.")
    
    # Extract LLM
    llm_candidates = parts[test_index + 1:]
    llm = None
    for candidate in llm_candidates:
        if candidate.startswith(('Qwen', 'llama')):
            llm = candidate
            break
    if llm is None:
        raise ValueError("LLM not found in the filename.")
    
    # Extract pattern
    pattern = '_'.join(parts[: contriever_index])
    
    return {
        'pattern': pattern,
        'contriever': parts[contriever_index],
        'task': task,
        'llm': llm
    }


if __name__ == '__main__':
    query = {
        'instruction': 'What is the capital of France?',
        'inputsentence': '',
        'response': 'The capital of France is Paris.',
        'example_text': 'Paris is the capital of France.'
    }
    print(get_train_prompt(query))