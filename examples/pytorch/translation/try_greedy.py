import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_en = "நான் பல்கலைக்கழகத்திற்கு செல்கிறேன்"
# article_en = "I'm going to university."
tokenizer.src_lang = "ta_IN"
encoded_en = tokenizer(article_en, return_tensors="pt")

output_sequence=[2]
generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                  num_beams=1, num_beam_groups=1, num_of_next_word_suggestions_list=4)

decoded_tokens_list=[]
for x, y in zip(generated_tokens["possible_next_wrods"][0],generated_tokens["possible_next_wrods_scores"][0]):
    decoded_token = tokenizer.decode(x, skip_special_tokens=True)
    s= str(x.item()) + ": " + decoded_token + " (" + str(y.item()) + ")"
    decoded_tokens_list.append(s)

print("Possible next tokens: {}".format(decoded_tokens_list))
next_tokens = int(input("Pick a next token ID: "))
output_sequence.append(next_tokens)
print("Partial output:- token IDs: {}, tokens: [{}]".format(output_sequence, tokenizer.decode(output_sequence, skip_special_tokens=True)))

output_sequence_eos_found=False

while output_sequence_eos_found==False:

    generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                                  num_beams=1, num_beam_groups=1, num_of_next_word_suggestions_list=4,
                                                  output_ids = output_sequence, current_length=generated_tokens["cur_len"],
                                                  custom_scores=generated_tokens["scores"],
                                                  custom_model_args=generated_tokens["custom_model_kwargs"],
                                                  custom_logit_processor=generated_tokens["custom_logit_processor"],
                                                  custom_encoder_attentions=generated_tokens.encoder_attentions,
                                                  custom_encoder_hidden_states=generated_tokens.encoder_hidden_states,
                                                  custom_decoder_attentions=generated_tokens.decoder_attentions,
                                                  custom_cross_attentions=generated_tokens.cross_attentions,
                                                  custom_decoder_hidden_states=generated_tokens.decoder_hidden_states,
                                                  custom_stopping_criteria=generated_tokens["custom_stopping_criteria"],
                                                  )

    decoded_tokens_list = []
    for x, y in zip(generated_tokens["possible_next_wrods"][0], generated_tokens["possible_next_wrods_scores"][0]):
        decoded_token = tokenizer.decode(x, skip_special_tokens=True)
        s = str(x.item()) + ": " + decoded_token + " (" + str(y.item()) + ")"
        decoded_tokens_list.append(s)

    print("Possible next tokens: {}".format(decoded_tokens_list))
    next_tokens = int(input("Pick a next token ID: "))
    output_sequence.append(next_tokens)
    print("Partial output:- token IDs: {}, tokens: [{}]".format(output_sequence, tokenizer.decode(output_sequence,
                                                                                                  skip_special_tokens=True)))

    if next_tokens==2:
        output_sequence_eos_found = True
