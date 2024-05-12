import sys
from time import time
from tqdm import tqdm
sys.path.insert(0, '../KGWatermark')
from extended_watermark_processor import WatermarkLogitsProcessor
from extended_watermark_processor import WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList
from transformers import GenerationConfig
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("allenai/c4", "realnewslike")

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", device_map = 'auto')

model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", device_map = 'auto')

# input_text = '''Repeat the following content: {}. Repitition:'''.format("(Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal")
# print(input_text)
# input_text = '''Repeat the following content: The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal marshes of the Northeastern and southern United States, and in Bermuda.[6] It belongs to the monotypic genus Malaclemys. It has one of the largest ranges of all turtles in North America, stretching as far south as the Florida Keys and as far north as Cape Cod.[7] The name 'terrapin' is derived from the Algonquian word torope.[8] It applies to Malaclemys terrapin in both British English and American English. The name originally was used by early European settlers in North America to describe these brackish-water turtles that inhabited neither freshwater habitats nor the sea. It retains this primary meaning in American English.[8] In British English, however, other semi-aquatic turtle species, such as the red-eared slider, might also be called terrapins. The common name refers to the diamond pattern on top of its shell (carapace), but the overall pattern and coloration vary greatly. The shell is usually wider at the back than in the front, and from above it appears wedge-shaped. The shell coloring can vary from brown to grey, and its body color can be grey, brown, yellow, or white. All have a unique pattern of wiggly, black markings or spots on their body and head. The diamondback terrapin has large webbed feet.[9] The species is'''

truncated_dataset = []
threshold = 70

print("Start Tokenization...")
tokenized_tensors = list(map(lambda s: tokenizer(s, return_tensors='pt'), dataset['validation']['text']))
print("End Tokenization...")

truncated_tokenized_tensors = []
decoded_text_inputs = []
for item in tqdm(tokenized_tensors):
    truncated_tokenized_tensors.append({'input_ids':item['input_ids'][:, :threshold].to(device), 
                                        'attention_mask': item['attention_mask'][:,:threshold].to(device)})
    
for item in tqdm(truncated_tokenized_tensors):
    # Assuming 'input_ids' are on the CPU for decoding
    input_ids = item['input_ids'][0].cpu().numpy()
    decoded_text_input = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded_text_inputs.append(decoded_text_input)


watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

output_text_list = []
output_text_normal_list = []
epoch = 2000

print(model.device)
print(truncated_tokenized_tensors[0]['input_ids'].device)



generation_config = GenerationConfig(
        max_new_tokens=80,        
        # do_sample = True,
        num_beams = 4,
        early_stopping = True,
        # temperature = 0.35,
        no_repeat_ngram_size = 4,
        # top_p=0.92
    )
epoch = 2000
for item in tqdm(truncated_tokenized_tensors[:epoch]):
    # start_time = time()

    
    output_tokens = model.generate(**item,
                                   logits_processor=LogitsProcessorList([watermark_processor]),
                                   generation_config = generation_config
                                  )
    output_tokens_normal = model.generate(**item,
                                   generation_config = generation_config
                                  )
    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked, the input/prompt is not
    output_tokens = output_tokens[:,item["input_ids"].shape[-1]:]
    output_tokens_normal = output_tokens_normal[:,item["input_ids"].shape[-1]:]
    
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    output_text_normal = tokenizer.batch_decode(output_tokens_normal, skip_special_tokens=True)[0]

    output_text_list.append(output_text)
    output_text_normal_list.append(output_text_normal)
    
    # end_time = time()
    # print(end_time - start_time)

import json
output_dict = {str(i): text for i, text in enumerate(output_text_list)}
output_normal_dict = {str(i): text for i, text in enumerate(output_text_normal_list)}
filename = 'output_texts.json'
filename_normal = 'output_normal_texts.json'
with open(filename, 'w') as f:
    json.dump(output_dict, f, indent=4)
with open(filename_normal, 'w') as f:
    json.dump(output_normal_dict, f, indent=4)
    

print(f"Data has been written to {filename}.")
print(f"Data has been written to {filename_normal}.")
