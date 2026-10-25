import os

cache_dir = os.environ.get('SUBICL_MODEL_CACHE_DIR', '/path/to/peft_models')
npy_dir = 'npy_files'
ragged_dir = 'ragged_datasets'

retriever_mappings = {
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'bmretriever': 'BMRetriever/BMRetriever-410M',
    'medcpt': 'ncbi/MedCPT-Query-Encoder',
}

pooling_mappings = {
    'bge-large': 'cls_pooling',
    'bmretriever': 'eos_pooling',
    'medcpt': 'cls_pooling',
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

task_contriever_coef_mappings = {
    'ner': {'bge-large': 1., 'bmretriever': 0.03, 'medcpt': 1.},
    're': {'bge-large': 1., 'bmretriever': 0.03, 'medcpt': 0.8},
    'tc': {'bge-large': 1., 'bmretriever': 0.02, 'medcpt': 1.},
}
