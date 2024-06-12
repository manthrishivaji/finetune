from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer,GenerationConfig
import torch
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig

# Choosing the device
device ='cuda' if torch.cuda.is_available() else 'cpu'

# Load the dataset 
huggingface_dataset_name = "knkarthick/dialogsum"
dataset =load_dataset(huggingface_dataset_name)

# Load the Model & Tokenizer 
model_name = "google/flan-t5-base"
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# check how many parameters does the mdoel have
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

## check the base model performance without finetuning 
# i= 10
# dialogue = dataset['test'][i]['dialogue']
# summary = dataset['test'][i]['summary']


# prompt = f"Summarize the following dialogue  {dialogue}  Summary:"


# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# output = tokenizer.decode(base_model.generate(input_ids, max_new_tokens=200)[0],skip_special_tokens=True)


# print(f"Input Prompt : {prompt}")
# print("--------------------------------------------------------------------")
# print("Human evaluated summary ---->")
# print(summary)
# print("---------------------------------------------------------------------")
# print("Baseline model generated summary : ---->")
# print(output)

# Function for structuring the data 
def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example
  
# Applying on the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# selecting the values for finetuning only which satifies  the index value == 100, some of rows for fine tuning would be 124
tokenized_datasets_1 = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)


# Setting the lora configurations
lora_config = LoraConfig(r=32,lora_alpha = 32, target_modules=["q","v"],
                         lora_dropout = 0.5, bias ="none", task_type  =TaskType.SEQ_2_SEQ_LM)

output_dir = f"./peft-dialogue-summary-training"

peft_model_train = get_peft_model(base_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model_train))

# setting configuration 
peft_training_args = TrainingArguments(
     output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=5,
 )
peft_trainer = Trainer(
    model=peft_model_train,
    args=peft_training_args,
    train_dataset=tokenized_datasets_1["train"],
    eval_dataset=tokenized_datasets_1["test"],

)
peft_trainer.train()

# saving model 
peft_model_path="./peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# Loading the saved model
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(peft_model_base,
                                       './peft-dialogue-summary-checkpoint-local',is_trainable=False)

# # Checking the model perfomance now after fine tuning
# peft_model_outputs = peft_model.generate(input_ids=input_ids, max_new_tokens=200)
# peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
 
# print(f"Input Prompt : {prompt}")
# print("--------------------------------------------------------------------")
# print("Human evaluated summary ---->")
# print(summary)
# print("---------------------------------------------------------------------")
# print("Baseline model generated summary : ---->")
# print(output)
# print("---------------------------------------------------------------------")
# print("Peft model generated summary : ---->")
# print(peft_model_text_output)



