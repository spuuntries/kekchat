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
    encoded_messages = [tokenizer.encode(
        message, return_tensors='pt') for message in messages]
    
    # Append the encoded messages to the chat history
    chat_history_ids = torch.cat(
        [encoded_messages[i] for i in range(len(encoded_messages))], dim=-1)
    
    # Generate a response
    chat_history_ids = model.generate(
        chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    response = tokenizer.decode(
        chat_history_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Return the response
    return response
