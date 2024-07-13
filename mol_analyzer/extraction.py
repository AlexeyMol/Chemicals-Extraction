from types import TracebackType
from .constants import Constants as constant
import subprocess
from collections import OrderedDict
from typing import List, Union, Dict
import nltk
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda


class Extractor:
    device = 'cuda' if cuda.is_available() else 'cpu'

    def __init__(self, hugger_name='AlexeyMol/mBERT_chemical_ner'):
        self.tokenizer = BertTokenizer.from_pretrained(hugger_name)
        self.id2label = constant.ID_TO_LABEL
        self.label2id = constant.LABEL_TO_ID
        self.model = BertForTokenClassification.from_pretrained(hugger_name,
                                                                num_labels=len(
                                                                    self.id2label),
                                                                id2label=self.id2label,
                                                                label2id=self.label2id)

        self.model.to(self.device)
        nltk.download('punkt')

    def __enter__(self):
        return self
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) ->None:
        pass
    
    def drop_duplicates(self, list_of_lists: List[List]) -> List[List]:
        """
        Remove duplicate lists from a list of lists.

        This function takes a list of lists and removes any duplicate lists, preserving the order of first occurrence.

        Args:
            list_of_lists (List[List]): A list of lists from which duplicates should be removed.

        Returns:
            List[List]: A list of lists with duplicates removed.
        """
        unique_list_of_lists = list(OrderedDict.fromkeys(
            tuple(lst) for lst in list_of_lists))
        unique_list_of_lists = [list(tup) for tup in unique_list_of_lists]

        return unique_list_of_lists

    def concatenate_lists(self, list_of_lists: List[List[Union[int, str]]]) -> List[List[Union[int, str]]]:
        """
        Concatenate contiguous sublists based on index values.

        This function takes a list of sublists where each sublist contains an integer start index, an integer end index, 
        and a string. It concatenates contiguous sublists where the end index of one sublist is immediately before 
        the start index of the next sublist, combining the indexes and concatenating the strings.

        Args:
            list_of_lists (List[List[Union[int, str]]]): A list of sublists to concatenate. Each sublist should contain 
                                                        an integer start index, an integer end index, and a string.

        Returns:
            List[List[Union[int, str]]]: A list of concatenated sublists.
        """
        concatenated_lists = []
        i = 0
        while i < len(list_of_lists):
            current_list = list_of_lists[i]
            concat_list = current_list
            while i < len(list_of_lists) - 1 and concat_list[1] == list_of_lists[i + 1][0] - 1:
                concat_list = [concat_list[0], list_of_lists[i + 1]
                               [1], concat_list[2] + ' ' + list_of_lists[i + 1][2]]
                i += 1
            concatenated_lists.append(concat_list)
            i += 1
        return concatenated_lists

    def extract_chemicals_from_text(self, text: str) -> Dict[str, Dict[int, List]]:
        """
        Extracts chemical entities (tokens labeled as 'B-chem' or 'I-chem') from the given text model.

        Parameters:
        - text (str): The input text to be processed.

        Returns:
        - dict: A dictionary containing the extracted chemical entities and their corresponding coordinates in the original text.
                The keys of the dictionary are the extracted chemical entities, and the values are dictionaries containing
                the coordinates (start and end indices) of each occurrence of the chemical entity in the text.
                Example:
                {
                    "chemical1": {1: [start_index1, end_index1], 2: [start_index2, end_index2], ...},
                    "chemical2": {1: [start_index1, end_index1], ...},
                    ...
                }

        """
        sentences = nltk.sent_tokenize(text)
        df = pd.DataFrame(sentences, columns=['sentence'])
        df_sentences = list(df.sentence)
        marked_result = []
        tokenized_sentences = []
        for sentence in df_sentences:

            inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            ids = inputs["input_ids"].to(self.device)
            mask = inputs["attention_mask"].to(self.device)

            outputs = self.model(ids, mask)
            logits = outputs[0]

        # Reshape logits and get predictions
            active_logits = logits.view(-1, self.model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

        # Get tokens and convert predictions
            tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [constant.ID_TO_LABEL[i] for i in flattened_predictions.cpu().numpy()]
            wp_preds = list(zip(tokens, token_predictions))

        # Initialize variables
            word_level_predictions = []
            current_word = ""
            current_prediction = ""
            tokenized_sentence = []

        # Iterate through wordpiece predictions
            for token, prediction in wp_preds:
                # Skip special tokens
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue

            # Handle wordpieces that start with '##'
                if token.startswith("##"):
                    current_word += token[2:]
                else:
                # End of previous word
                    if current_word:
                        word_level_predictions.append((current_word, current_prediction))
                        tokenized_sentence.append(current_word)
                        current_word = ""
                        current_prediction = ""

                    current_word = token
                    current_prediction = prediction

        # Append the last word's prediction
            if current_word:
                word_level_predictions.append((current_word, current_prediction))
                tokenized_sentence.append(current_word)

            tokenized_sentences.append(tokenized_sentence)

            matching_sentence_with_tags = [pred for _, pred in word_level_predictions]
            marked_result.append(matching_sentence_with_tags)

        df['marked_results'] = marked_result

        df['tokenized_sentence'] = tokenized_sentences
        # writing to tmp .ann file
        with open('ann.ann', 'w') as file:
            i = 1
            for row in df.iterrows():
                tokens = row[1]['marked_results']
                words = row[1]['tokenized_sentence']
                res = []
                for w, t in zip(words, tokens):
                    if t == 'B-chem' or t == 'I-chem' or t == 'S-chem' or t == 'L-chem':
                        res.append(w)
                for value in res:
                    word_index = row[1]['sentence'].find(value)
                    sent_start = text.find(row[1]['sentence'])
                    result_index_start = sent_start + word_index
                    result_index_end = result_index_start + len(value)
                    if value in constant.NOISE or len(value) <=1:
                        continue
                    try:
                        int(value)
                    except:
                        try:
                            float(value)
                        except:
                            file.write(f'T{i}\tCHEMICAL {result_index_start} {result_index_end}\t{value}\n')
                            i += 1
        # postprocessing
        i = 1
        res = []
        with open(f'ann.ann', 'r') as file:
            for line in file:
                word = line[line.rfind('\t') + 1: ]
                line = line[line.find(' ') + 1: ]
                line = line[: line.find('\t')] + ' ' + word[: -1]
                row = line.split(' ')
                row[0] = int(row[0])
                row[1] = int(row[1])
                res.append(row)
        res = sorted(res, key=lambda x: x[0])
        res = self.concatenate_lists(res)
        res = self.drop_duplicates(res)
        with open(f'ann.ann', 'w') as file:
            for el in res:
                file.write(f'T{i}\tCHEMICAL {el[0]} {el[1]}\t{el[2]}\n')
                i += 1
        json_res = {}
        for row in res:
            if row[2] in json_res.keys():
                j = len(json_res[row[2]]) + 1
                json_res[row[2]][j] = [row[0], row[1]]
            else:
                json_res[row[2]] = {1: [row[0], row[1]]}
        subprocess.run(['rm', 'ann.ann'])
        return json_res