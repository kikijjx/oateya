import torch
import json
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset

tokenizer = T5Tokenizer.from_pretrained("t5-small")

class TranslationDataset(Dataset):
    def __init__(self, data, src_lang, tgt_lang):
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source = self.data[idx][self.src_lang]
        target = self.data[idx][self.tgt_lang]
        return {
            "input_ids": torch.tensor(source),
            "labels": torch.tensor(target)
        }

def load_and_split_data(data_file, src_lang, tgt_lang, test_size=0.1):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    train_data, eval_data = train_test_split(data, test_size=test_size, random_state=42)
    
    train_dataset = TranslationDataset(train_data, src_lang, tgt_lang)
    eval_dataset = TranslationDataset(eval_data, src_lang, tgt_lang)
    
    return train_dataset, eval_dataset

def train_model(train_file, output_dir, src_lang, tgt_lang):
    train_dataset, eval_dataset = load_and_split_data(train_file, src_lang, tgt_lang)
    
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model trained and saved to {output_dir}")

if __name__ == "__main__":
    train_model("tokenized_data.json", "model_ru_en", "russian", "english")
    train_model("tokenized_data.json", "model_en_ru", "english", "russian")
