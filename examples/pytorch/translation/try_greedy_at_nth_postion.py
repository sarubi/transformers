from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

input_sentence = "நான் பல்கலைக்கழகத்திற்கு செல்கிறேன்"
# output_sentence = "I'm going to university."
tokenizer.src_lang = "ta_IN"
encoded_en = tokenizer(input_sentence, return_tensors="pt")
print("Input tokens: {}".format(input_sentence))

output_sequence=[2, 250004, 87, 444, 7730, 47]
print("Partial output:- token IDs: {}, tokens: [{}]".format(output_sequence, tokenizer.decode(output_sequence, skip_special_tokens=True)))
# output_sequence=[2, 250004, 87, 444, 7730, 47, 152363, 2]
generated_tokens = model.generate_with_suggestion(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],return_dict_in_generate=True, output_scores=True,
                                  num_beams=1, num_beam_groups=1, num_of_next_word_suggestions_list=3,output_ids = output_sequence)

decoded_tokens_list=[]
for x, y in zip(generated_tokens["possible_next_wrods"][0],generated_tokens["possible_next_wrods_scores"][0]):
    decoded_token = tokenizer.decode(x, skip_special_tokens=True)
    s= str(x.item()) + ": " + decoded_token + " (" + str(y.item()) + ")"
    decoded_tokens_list.append(s)

print("Possible next tokens: {}".format(decoded_tokens_list))
next_tokens = int(input("Pick a next token ID: "))
output_sequence.append(next_tokens)
print("Partial output:- token IDs: {}, tokens: [{}]".format(output_sequence, tokenizer.decode(output_sequence, skip_special_tokens=True)))
