import os
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from transformers import AutoConfig
import torch
import numpy

# Create a cache directory to store pretrained model.
cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_name_or_path = "gpt2"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)


num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer

from transformers import AutoTokenizer

EXAMPLE_Text = ['best hotel in bay area', 'here is an example of gpt2 model', 'I like walking with my dog']

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    #okenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def preprocess(tokenizer, prompt_text):    
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
       
    return input_ids, attention_mask, position_ids, empty_past


def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                 past_sequence_length=past[0].size(3),
                                                 sequence_length=input_ids.size(1),
                                                 config=config)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past,
                                               output_buffers, output_shapes)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes,
                                                            return_numpy=False)
    return outputs

def test_generation(tokenizer, session, input_text, num_tokens_to_produce = 30):

    eos_token_id = tokenizer.eos_token_id
    
    input_ids, attention_mask, position_ids, past = preprocess(tokenizer, input_text)
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    generated_text = []

    for step in range(num_tokens_to_produce):
        outputs = inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past)

        next_token_logits = outputs[0][:, -1, :]

        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)    
        position_ids = (position_ids[:,-1] + 1).reshape(batch_size,1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)    

        past = []
        for i in range(num_layer):
            past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()
            past.append(past_i.to(device))

        if torch.all(has_eos):
            break

    for i, output in enumerate(all_token_ids):
        text = tokenizer.decode(output, skip_special_tokens=True)
        #print("------------")
        #print(text)
        generated_text.append(text)


import onnxruntime

tokenizer = get_tokenizer('gpt2', cache_dir)

onnx_model_path = 'models/gpt2.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

#TODO input to the endpoint
prompts = ['best hotel in bay area', 'here is an example of gpt2 model']

#TODO output from the endpoint
generated_text = test_generation(tokenizer, session, prompts)

print(f'Generated text = {generated_text}') 



