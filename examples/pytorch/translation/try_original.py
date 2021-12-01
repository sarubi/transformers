import torch

from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch import nn


# model = AutoModel.from_pretrained("facebook/mbart-large-50")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_en = "நான் பல்கலைக்கழகத்திற்கு செல்கிறேன்"
# article_en = "I'm going to university."

tokenizer.src_lang = "ta_IN"
encoded_en = tokenizer(article_en, return_tensors="pt")

generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
 num_beams=1)

# num_beam_groups=1,do_sample=False,num_return_sequences=1
x=tokenizer.batch_decode(generated_tokens["sequences"], skip_special_tokens=True)
# x=tokenizer.batch_decode(generated_tokens[0], skip_special_tokens=True)
print("Final output sequence: {}".format(x))