# character_interaction_graph/rewriting.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def rewrite_with_t5_base(text):
    """
    Reescreve o texto definindo claramente sujeito, verbo e objeto utilizando o modelo T5.
    """
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    input_text = "Summarize all the character-character interactions in the text to clearly define the subject, verb, and object: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=512, num_return_sequences=1)
    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten_text

def rewrite_with_t5(text):
    """
    Reescreve o texto utilizando um pipeline de text-to-text com o modelo FLAN-T5.
    """
    from transformers import pipeline
    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    return pipe("Summarize the text in character - verb - character form: " + text)[0]['generated_text']

def summarize_t5(text):
    """
    Resume o texto utilizando o modelo T5 pequeno, destacando interações no formato 'personagem - verbo - personagem'.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    inputs = tokenizer.batch_encode_plus(
        ["Summarize the text in character - verb - character form: " + text],
        max_length=1024,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    outputs = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary
