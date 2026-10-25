import os
import torch
# import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoModel, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from glob import glob


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    lm_int8_enable_fp32_cpu_offload=True,
)


def get_retrieve_model(model_name_or_path, cache_dir=None):
    model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    return model, tokenizer


def get_tokenizer(model_name_or_path, padding_side):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    except:
        print('Use LlamaTokenizer...')
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    # print(tokenizer.pad_token, tokenizer.eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_latest_checkpoint(base_dir, pattern):
    all_folders = glob(os.path.join(base_dir, pattern))
    if len(all_folders) == 0:
        raise Exception("No checkpoints found in {}".format(base_dir))
    
    prefix = os.path.basename(all_folders[0]).split('-')[0]
    steps = [int(os.path.basename(folder).split('-')[1]) for folder in all_folders]
    return os.path.join(base_dir, prefix + '-' + str(max(steps)))
    # if os.path.exists(best_model_folder):
    #     os.rename(best_model_folder, new_folder_path)


def get_model_and_tokenizer(model_name_or_path, padding_side='right'):
    model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map='auto',
                )
    # model = nn.DataParallel(model)
    # print(model.config.use_sliding_window)
    return model, get_tokenizer(model_name_or_path, padding_side)


def get_sft_model_and_tokenizer(base_model, peft_model_path, padding_side='right'):
    latest_dir = get_latest_checkpoint(peft_model_path, 'checkpoint-*')
    print('Peft Model Dir loaded: ', latest_dir)
    model = PeftModel.from_pretrained(
                            base_model,
                            latest_dir,
                            torch_dtype=torch.float16,
                            device_map='auto',
                        )
    return model, get_tokenizer(latest_dir, padding_side)


def get_sft_trainer(train_data,
                    test_data,
                    base_model,
                    tokenizer,
                    args,
    ):
    # if base_model is None or tokenizer is None:
    #     base_model, tokenizer = get_model_and_tokenizer(llm_name_or_path, 'right')
    
    # if "mamba-2.8b-hf" in llm_name_or_path:
    #     qlora_config = LoraConfig(
    #         r=8,
    #         target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    #         task_type="CAUSAL_LM",
    #         bias="none"
    #     )
    # else:
    qlora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, qlora_config)
    base_model_state_dict = base_model.state_dict()
    # print(base_model_state_dict.keys())
    # for name, param in model.named_parameters():
    #     pattern_name = name.replace('base_model.model.model', 'model')
    #     if pattern_name in base_model_state_dict:
    #         base_param_device = base_model_state_dict[pattern_name].device
    #         if param.device != base_param_device:
    #             param.data = param.data.to(base_param_device)
    #             if param.grad is not None:
    #                 param.grad = param.grad.to(base_param_device)
    #     else:
    #         print(f"Parameter {name} not found in base_model state_dict.")

    return SFTTrainer(
            model,
            train_dataset=train_data,
            eval_dataset=test_data,
            args=SFTConfig(
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    per_device_eval_batch_size=args.per_device_eval_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.learning_rate,
                    max_steps=args.max_steps,
                    save_steps=args.save_steps,
                    eval_steps=args.eval_steps,
                    eval_strategy=args.eval_strategy,
                    max_grad_norm=0.3,
                    warmup_ratio=0.03,
                    output_dir=args.lora_dir,
                    optim="paged_adamw_8bit",
                    fp16=True,
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    dataset_text_field="text",
                    max_seq_length=args.max_seq_length,
                    # label_names=["labels"],
                ),
            tokenizer=tokenizer,
            # peft_config=qlora_config,
        )
