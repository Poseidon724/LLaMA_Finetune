# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
# pip install transformers
# pip install matplotlib
# pip install torch
# pip install accelerate
# pip install peft
# pip install datasets
# pip install wandb

import os
import transformers
import wandb
from datetime import datetime
import torch
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig

"""Function to load input-output pairs from a text file"""

def load_input_output_pairs(file_path):
    input_data = []
    output_data = []
    with open(file_path, "r") as file:
        input_line = None
        output_line = None
        for line in file:
            if line.startswith("Input: "):
                input_line = line[len("Input: "):].strip()  # Extract the input data
            elif line.startswith("Output: "):
                output_line = line[len("Output: "):].strip()  # Extract the output data
                # If both input and output are present, store them
                if input_line is not None and output_line is not None:
                    input_data.append(input_line)
                    output_data.append(output_line)
                    # Reset input_line and output_line for next pair
                    input_line = None
                    output_line = None
    return input_data, output_data

"""Load input-output pairs from the text file"""

input_data, output_data = load_input_output_pairs("/content/mistral_data.txt")

"""Create a dataset from the input-output pairs"""

combined_data = [{"input": input_text, "output": output_text} for input_text, output_text in zip(input_data, output_data)]
# custom_dataset = Dataset.from_dict(combined_data)

"""Create a DataLoader for batching and iterating over the dataset"""

batch_size = 32
# dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(combined_data, batch_size=batch_size, shuffle=True)

"""Set up the Accelerator"""

# from transformers import BertConfig
# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=transformers.FSDBertConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=transformers.FSDBertConfig(offload_to_cpu=True, rank0_only=False),
# )
# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
# from transformers import BertConfig

# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=BertConfig(offload_to_cpu=True, grad_checkpointing=True),
#     optim_state_dict_config=BertConfig(offload_to_cpu=True, grad_checkpointing=True),
# )

"""Set up logging with Weights & Biases"""

# wandb.login()
# wandb_project = "mistral-finetune"
# if len(wandb_project) > 0:
#     os.environ["WANDB_PROJECT"] = wandb_project

"""Function to format prompts"""

def formatting_func(example):
    return f"### Give a discourse simplification graph for: {example} where the usage is for legal documents"

"""Load Mistral model"""

# # Define the environment variable name
# TOKEN_ENV_VAR = "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"
# # Get the token from the environment variable
# hf_token = os.getenv(TOKEN_ENV_VAR)
# # Use hf_token in your code

# base_model_id = "mistralai/Mistral-7B-v0.1"
# model = AutoModelForCausalLM.from_pretrained(base_model_id)

# import os
# from transformers import AutoModelForCausalLM

# Define the environment variable name
# TOKEN_ENV_VAR = "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"
os.environ["HF_HOME"] = "false"

# Get the token from the environment variable
# hf_token = os.getenv(TOKEN_ENV_VAR)

# Use hf_token in your code
base_model_id = "mistralai/Mistral-7B-v0.1"
# model = AutoModelForCausalLM.from_pretrained(base_model_id, auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(base_model_id)

"""Tokenization"""

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

"""Tokenize dataset"""

max_length = 512
tokenized_dataset = combined_data.map(
    lambda example: tokenizer(
        formatting_func(example),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
)

"""Set up LoRA"""

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model, config)

"""Print trainable parameters"""

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params} || All params: {all_params} || Trainable %: {100 * trainable_params / all_params}"
    )

print_trainable_parameters(model)

"""Set up Trainer"""

project = "legal-finetune"
base_model_name = "mistral"
run_name = f"{base_model_name}-{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
output_dir = f"./{run_name}"
trainer = transformers.Trainer(
    model=model,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        # report_to="wandb",
        run_name=run_name
    ),
    train_dataset=tokenized_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    eval_dataset=tokenized_dataset,
    accelerator=accelerator,
)

"""Train the model"""

trainer.train()

"""Optional: Evaluate the trained model on a separate validation set<br>
Optional: Try the trained model on some input examples
"""