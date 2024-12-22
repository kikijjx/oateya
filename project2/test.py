import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import TER
from tqdm import tqdm

model = T5ForConditionalGeneration.from_pretrained("model_en_ru")
tokenizer = T5Tokenizer.from_pretrained("model_en_ru")

data = pd.read_csv("sentences.csv").sample(1000, random_state=42)

def generate_predictions(model, tokenizer, texts, max_new_tokens=50, batch_size=8):
    model.eval()
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"):
        batch = texts[i:i + batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        decoded_batch = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(decoded_batch)
    return predictions

source_texts = data['english'].tolist()
reference_texts = [[ref] for ref in data['russian'].tolist()]

predictions = generate_predictions(model, tokenizer, source_texts)

bleu_score = corpus_bleu(reference_texts, [pred.split() for pred in predictions])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(ref[0], pred) for ref, pred in zip(reference_texts, predictions)]
average_rouge = {
    'rouge1': sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
    'rouge2': sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores),
    'rougeL': sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores),
}

meteor_scores = [meteor_score(ref, pred) for ref, pred in zip(reference_texts, predictions)]
average_meteor = sum(meteor_scores) / len(meteor_scores)

ter_metric = TER()
ter_score = ter_metric.corpus_score(predictions, [ref[0] for ref in reference_texts]).score

print(f"BLEU: {bleu_score:.4f}")
print(f"ROUGE: {average_rouge}")
print(f"METEOR: {average_meteor:.4f}")
print(f"TER: {ter_score:.4f}")
