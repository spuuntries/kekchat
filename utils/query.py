from pydoc import doc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "spuun/kekbot-beta-3-medium", cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained(
    "spuun/kekbot-beta-3-medium", cache_dir="./cache")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = model.to(device)


def query(input_string: str) -> str:
    # Function to query the chatbot,
    # takes a string as input (of all the previous chat history, which will be tokenized)
    # and returns a string as output (the chatbot's response)

    # Split the input string into messages
    messages = input_string.split("\n")

    # Encode the messages
    encoded_messages = torch.cat([tokenizer.encode(
        message + tokenizer.eos_token, return_tensors='pt') for message in messages], dim=-1)

    # Generate a response
    generated = model.generate(
        encoded_messages, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.18,
        repetition_penalty_range=512,
        repetition_penalty_slope=3.33,
        do_sample=True,
        temperature=0.85)

    # Decode the response
    response = tokenizer.decode(
        generated[:, encoded_messages.shape[-1]:][0], skip_special_tokens=True)

    # Return the response
    return response
