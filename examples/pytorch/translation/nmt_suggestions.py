from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_en = "நான் பல்கலைக்கழகத்திற்கு செல்கிறேன்"
# article_en = "I'm going to university."
tokenizer.src_lang = "ta_IN"
encoded_en = tokenizer(article_en, return_tensors="pt")

generated_tokens = model.generate_with_suggestion_completion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                  num_beams=4,num_return_sequences=1)

print("sequence ids: {} \ttoken:{}".format(generated_tokens["sequences"], tokenizer.batch_decode(generated_tokens["sequences"], skip_special_tokens=True)))
print("*************************")

output_sequence=[ 2, 87, 444, 7730, 47]
print("Partial sequence: {} \n{}".format(output_sequence, tokenizer.batch_decode(output_sequence, skip_special_tokens=True)))

generated_tokens = model.generate_with_suggestion_completion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                              num_beams=4,num_return_sequences=4,
                                              output_ids = output_sequence,
                                              )

print("possible sentence completion: {} \ttoken:{}".format(generated_tokens["sequences"], tokenizer.batch_decode(generated_tokens["sequences"], skip_special_tokens=True)))

print("Possible next tokens:")

for id in generated_tokens["possible_next_wrods"][0]:
    token=tokenizer.decode(id, skip_special_tokens=True)
    print("\tID: {} - {}".format(id,token))

next_tokens = int(input("Pick a next token ID: "))
