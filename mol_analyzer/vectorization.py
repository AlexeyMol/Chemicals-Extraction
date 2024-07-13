from typing import Union
import torch
from transformers import BertTokenizer, BertModel
from torch import cuda


def vectorize_text(
    text: str, 
    to_insert_into_postgres: bool = False,
    model:any = None,
    tokenizer:any = None,
    device:any = None
) -> Union[str, torch.Tensor]:
    """
    Converts input text to a vector using a pre-defined BERT model.

    This function tokenizes the input text and processes it through a BERT model to 
    obtain a sentence embedding. The sentence embedding is calculated as the mean 
    of the last hidden state of the BERT model. If the input text exceeds the model's 
    maximum token length (512 tokens), a ValueError is raised.

    Args:
        text (str): The input text to be converted to a vector.
        to_insert_into_postgres (bool, optional): If True, returns the sentence embedding 
            as a stringified list suitable for insertion into PostgreSQL. If False, 
            returns the raw tensor of the sentence embedding. Defaults to False.

    Returns:
        Union[str, torch.Tensor]: The sentence embedding. Returns a stringified list if 
        `to_insert_into_postgres` is True, otherwise returns a raw tensor.

    Raises:
        ValueError: If the input text exceeds the model's maximum token length.

    """
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True)  # max_length=512
    
    # Check if tokenized input length exceeds 512 tokens
    if inputs['input_ids'].size(1) > 512:
        raise ValueError('Your text is longer than the maximum model input length (512 tokens).\nTry to divide it into batches.')
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
    
    # Compute the mean of the last hidden state to get the sentence embedding
    sentence_embedding = torch.mean(last_hidden_state[0], dim=0)
    
    if to_insert_into_postgres:
        # return str for postgres
        return str(sentence_embedding.tolist())
    # return raw vector
    return sentence_embedding