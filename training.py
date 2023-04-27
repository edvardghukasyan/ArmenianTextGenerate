import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class CustomTextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.lines = self.load_lines()

    def load_lines(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]
        return lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.tokenizer(self.lines[idx], return_tensors="pt")

vocab_size = 50000
num_layers = 12
num_attention_heads = 16
hidden_dim = 768
max_seq_length = 512
dropout_rate = 0.1
learning_rate = 2e-5
batch_size = 8
num_epochs = 3
dataset_file = "armenian_dataset.txt"
test_dataset_file = "armenian_test_dataset.txt"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", max_len=max_seq_length)
tokenizer.add_special_tokens({"additional_special_tokens": ["<sep>"]})

train_dataset = CustomTextDataset(dataset_file, tokenizer)
test_dataset = CustomTextDataset(test_dataset_file, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

config = GPT2Config(
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_attention_heads,
    n_embd=hidden_dim,
    n_ctx=max_seq_length,
    n_positions=max_seq_length,
    dropout=dropout_rate,
)

model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained("fine-tuned")
tokenizer.save_pretrained("fine-tuned")