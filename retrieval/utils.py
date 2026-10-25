import json
import ast
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Find most similar indices
def find_most_similar_indices(tensor_a, tensor_b=None):
    """
    Find_most_similar_indices(tensor_a, tensor_b=None) -> List[List[int]]
    
    If tensor_b is None, find the most similar indices in tensor_a (ignoring itself).
    If tensor_b is provided, find the most similar indices in tensor_b for each tensor_a.
    """  
    same_tensor = tensor_b is None
    tensor_b = deepcopy(tensor_a) if same_tensor else tensor_b

    tensor_a = tensor_a.unsqueeze(1)  # shape: (n, 1, m)
    tensor_b = tensor_b.unsqueeze(0)  # shape: (1, n, m)
    print(tensor_a.shape, tensor_b.shape)
    sim_matrix = F.cosine_similarity(tensor_a, tensor_b, dim=2)
    # if same_tensor:
    #     cos_sim_matrix.fill_diagonal_(-float('inf'))
    sim_ranking = sim_matrix.argsort(dim=1, descending=True).tolist()
    print()
    return sim_ranking


def retrieve(example_file,
             input_file,
             output_file,
             model,
             tokenizer,
             group_key="response",
             max_length=512,
    ):
    grouped_data = defaultdict(list)
    ## all training data consists of retriever corpus
    all_examples = []
    with open(example_file, 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin.readlines()):
            data = json.loads(line.strip())
            example = data['inputsentence'] + '\nResponse:' + data['response']
            all_examples.append(example)
            grouped_data[data[group_key]].append(idx)
    
    print(f"Total {len(grouped_data)} classes!!")
    
    grouped_embeds = {}
    for key in grouped_data.keys():
        print(len(key))
        examples_to_tokenize = [all_examples[i] for i in grouped_data[key]]
        inputs = tokenizer(
                examples_to_tokenize, truncation=True, padding=True, return_tensors='pt', max_length=512)
        outputs = model(**inputs)
        embeds = mean_pooling(outputs[0], inputs['attention_mask'])
        grouped_embeds[key] = embeds
        break
    return


    all_data = []
    # input_examples = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for index, line in tqdm(enumerate(lines), desc="Processing input file", total=len(lines), unit="line", ncols=80):
            data = json.loads(line)
            example = data['inputsentence'] + '\nResponse:' + data['response']
            inputs = tokenizer(
                    [example],
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=512,
                )
            outputs = model(**inputs)
            input_embeds = mean_pooling(outputs[0], inputs['attention_mask'])
            
            data["example_index"] = []
            for key in grouped_embeds:
                sim_ranking = find_most_similar_indices(input_embeds, grouped_embeds[key])
                data["example_index"].append(similarity_ranking[0][0] if index!=similarity_ranking[0][0] else similarity_ranking[0][1])
            # input_examples.append(example)
            all_data.append(data)
            
    print("saving data....")
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(all_data, fout, indent=4)
        fout.write("\n")
    print("Data saved!")


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def max_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), float('-inf'))
    # mask_expanded = mask.unsqueeze(-1).expand_as(token_embeddings).bool()
    # masked_embeddings = token_embeddings.masked_fill(~mask_expanded, float('-inf'))
    sentence_embeddings, _ = torch.max(token_embeddings, dim=1)
    return sentence_embeddings

def last_token_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding

def encode_batch_quries(query_list, model, tokenizer, pooling, max_length, batch_size):
    assert type(query_list) == list
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    all_embeds = []
    start_idx = 0
    while start_idx < len(query_list):
        end_idx = min(start_idx + batch_size, len(query_list))
        batch_sentences = query_list[start_idx: end_idx]
        if pooling == 'eos_pooling': # for bmretriever
            # print(tokenizer.eos_token)
            batch_sentences = [sent + tokenizer.eos_token for sent in batch_sentences]
        inputs = tokenizer(
                batch_sentences, truncation=True, padding=True, return_tensors='pt', max_length=max_length)
        model, inputs = model.to(device), inputs.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            # print(inputs['input_ids'])

            ## outputs[0] is also outputs.last_hidden_state
            if pooling == 'mean_pooling':
                embeds = mean_pooling(outputs[0], inputs['attention_mask']) # (batch_size, 768)
            elif pooling == 'max_pooling':
                embeds = max_pooling(outputs[0], inputs['attention_mask']) # (batch_size, 768)
            elif pooling == 'eos_pooling':
                # embeds = outputs[0][:, -1, :]
                embeds = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            elif pooling == 'cls_pooling':
                embeds = outputs[0][:, 0, :]
            else:
                raise ValueError(f"Pooling method {pooling} not supported!")
            # print(embeds, embeds.shape)
            embeds = embeds.cpu().numpy()
        all_embeds.extend([embeds[i] for i in range(len(embeds))])
        start_idx += batch_size
    
    all_embeds = np.array(all_embeds)
    assert all_embeds.shape[0] == len(query_list)
    return all_embeds


def generate_sent_embedding(example_file,
                            output_file,
                            model,
                            tokenizer,
                            pooling,
                            max_length=512,
                            batch_size=32,
    ):
    all_queries = []
    flag = False
    with open(example_file, 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin.readlines()):
            data = json.loads(line.strip())
            query = data['inputsentence']
            # if query.startswith('[') and query.endswith(']'):
            #     flag = True
            #     query = ' '.join(ast.literal_eval(query))
            all_queries.append(query)
    print(f"Total {len(all_queries)} sentences!!, convert list to string: {flag}")
    
    embedding_matrix = encode_batch_quries(all_queries, model, tokenizer, pooling, max_length, batch_size)
    np.save(output_file, embedding_matrix)
    print("Embedding saved!")
    return


if __name__ == "__main__":
    query = "['In', 'the', 'yeast', 'two', '-', 'hybrid', 'assays', ',', 'PNRC', 'interacted', 'with', 'the', 'orphan', 'receptors', 'SF1', 'and', 'ERRalpha1', 'in', 'a', 'ligand', '-', 'independent', 'manner', '.']"
    print(' '.join(ast.literal_eval(query)))