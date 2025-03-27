# character_interaction_graph/rewriting.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import torch



def summarize_t5(text):
    """
    Summarize text using a T5-small model, highlighting character interactions.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    prompt = "Summarize the text in character - verb - character form: " + text

    inputs = tokenizer.batch_encode_plus(
        [prompt],
        max_length=10240,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    outputs = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=500,
        early_stopping=True
    )

    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summarized_text

