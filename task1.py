import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

ARTICLE_TO_SUMMARIZE = '''The win by Kelly, who was elected in 2020 to fill the term of the late GOP Sen. John McCain, 
capped a string of victories for Democrats on Friday night as ballots continued to be painstakingly tallied in the West. 
Kelly’s defeat of venture capitalist Blake Masters, who had echoed former President Donald Trump’s lies about the 2020 election, 
marked yet another rejection by voters of a Trump-backed candidate who Democrats portrayed as an extremist.'''
inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

# Global attention on the first token (cf. Beltagy et al. 2020)
global_attention_mask = torch.zeros_like(inputs)
global_attention_mask[:, 0] = 1

# Generate Summary
summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))