import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import numpy as np

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_en = "நான் பல்கலைக்கழகத்திற்கு செல்கிறேன்"
# article_en = "I'm going to university."
tokenizer.src_lang = "ta_IN"
encoded_en = tokenizer(article_en, return_tensors="pt")

output_sequence=[2]

generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                  num_beams=3, num_beam_groups=1)

print("Possible next tokens: {}".format(generated_tokens["possible_next_wrods"]))
next_tokens = int(input("Pick a next token ID: "))
output_sequence.append(next_tokens)

output=torch.tensor([[     2, 250004],
        [     2, 250004],
        [     2, 250004]])

generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                                  num_beams=3, num_beam_groups=1,
                                                  output_ids = output, current_length=generated_tokens["cur_len"],
                                                  custom_beam_scorer=generated_tokens["custom_beam_scorer"],
                                                  custom_scores=generated_tokens["scores"],
                                                  custom_beam_scores=generated_tokens["beam_scores"],
                                                  custom_model_args=generated_tokens["custom_model_kwargs"],
                                                  custom_logit_processor=generated_tokens["custom_logit_processor"],
                                                  custom_encoder_attentions=generated_tokens.encoder_attentions,
                                                  custom_encoder_hidden_states=generated_tokens.encoder_hidden_states,
                                                  custom_decoder_attentions=generated_tokens.decoder_attentions,
                                                  custom_cross_attentions=generated_tokens.cross_attentions,
                                                  custom_decoder_hidden_states=generated_tokens.decoder_hidden_states,
                                                  custom_stopping_criteria=generated_tokens["custom_stopping_criteria"],
                                                  )

print("Possible next tokens: {}".format(generated_tokens["possible_next_wrods"]))
# num1 = int(input("Pick a next token ID: "))
# next_tokens = num1
num1=[2,250004,87]

output=torch.tensor([[     2, 250004, 87],
        [     2, 250004, 87],
        [     2, 250004, 87]])

generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                                  num_beams=3, num_beam_groups=1,
                                                  output_ids = output, current_length=generated_tokens["cur_len"],
                                                  custom_beam_scorer=generated_tokens["custom_beam_scorer"],
                                                  custom_scores=generated_tokens["scores"],
                                                  custom_beam_scores=generated_tokens["beam_scores"],
                                                  custom_model_args=generated_tokens["custom_model_kwargs"],
                                                  custom_logit_processor=generated_tokens["custom_logit_processor"],
                                                  custom_encoder_attentions=generated_tokens.encoder_attentions,
                                                  custom_encoder_hidden_states=generated_tokens.encoder_hidden_states,
                                                  custom_decoder_attentions=generated_tokens.decoder_attentions,
                                                  custom_cross_attentions=generated_tokens.cross_attentions,
                                                  custom_decoder_hidden_states=generated_tokens.decoder_hidden_states,
                                                  custom_stopping_criteria=generated_tokens["custom_stopping_criteria"],
                                                  )


print("Possible next tokens: {}".format(generated_tokens["possible_next_wrods"]))



num1=[2,250004,87,25]

output=torch.tensor([[     2, 250004, 87,25],
        [     2, 250004, 87,25],
        [     2, 250004, 87,25]])

generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                                  num_beams=3, num_beam_groups=1,
                                                  output_ids = output, current_length=generated_tokens["cur_len"],
                                                  custom_beam_scorer=generated_tokens["custom_beam_scorer"],
                                                  custom_scores=generated_tokens["scores"],
                                                  custom_beam_scores=generated_tokens["beam_scores"],
                                                  custom_model_args=generated_tokens["custom_model_kwargs"],
                                                  custom_logit_processor=generated_tokens["custom_logit_processor"],
                                                  custom_encoder_attentions=generated_tokens.encoder_attentions,
                                                  custom_encoder_hidden_states=generated_tokens.encoder_hidden_states,
                                                  custom_decoder_attentions=generated_tokens.decoder_attentions,
                                                  custom_cross_attentions=generated_tokens.cross_attentions,
                                                  custom_decoder_hidden_states=generated_tokens.decoder_hidden_states,
                                                  custom_stopping_criteria=generated_tokens["custom_stopping_criteria"],
                                                  )


print("Possible next tokens: {}".format(generated_tokens["possible_next_wrods"]))




# generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
# num_beams=3, num_beam_groups=1,do_sample=False,num_return_sequences=1)


# print(generated_tokens.scores)
# print(generated_tokens.sequences_scores)
# print(generated_tokens[0])
# print(generated_tokens[1])
# print(generated_tokens[2])

# x=tokenizer.batch_decode(generated_tokens["sequences"], skip_special_tokens=True)
# x=tokenizer.batch_decode(generated_tokens[0], skip_special_tokens=True)
# print("Final output sequence: {}".format(x))

# score = torch.nn.functional.softmax(generated_tokens.sequences_scores,dim=-1)
# print(score)

# return_dict_in_generate
