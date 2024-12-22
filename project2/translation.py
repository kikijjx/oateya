from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(model_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def translate(text, model, tokenizer, src_lang, tgt_lang):
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_en_ru, tokenizer_en_ru = load_model("model_en_ru")
    print(translate("there is a light", model_en_ru, tokenizer_en_ru, "english", "russian"))

