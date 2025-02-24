import os
import torch
import logging
from transformers import AutoTokenizer, BertTokenizer


class MolecularTokenizer:
    """
    Tokenizer for molecular SMILES sequences
    """
    def __init__(self, args):
        """
        Args:
            max_length (int): Maximum sequence length
            model_name (str): Pre-training model name
        """
        self.max_length = args.max_seq_length
        self.model_path = args.bert_model_path
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.model_path, exist_ok=True)

        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Successfully loaded tokenizer from {self.model_path}")
        except Exception as e:
            print(f"Error loading tokenizer with AutoTokenizer: {e}")
            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                print("Successfully loaded tokenizer with BertTokenizer")
            except Exception as e:
                print(f"Error loading tokenizer with BertTokenizer: {e}")
                raise ValueError(f"Failed to load tokenizer from {self.model_path}")

    def encode(self, smiles_list, padding=True):
        try:
            encoded = self.tokenizer.batch_encode_plus(
                smiles_list,
                padding='max_length' if padding else False,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            assert isinstance(input_ids, torch.Tensor), "input_ids must be a tensor"
            assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a tensor"
            assert input_ids.size(1) == self.max_length, f"Unexpected sequence length: {input_ids.size(1)}"
            return input_ids, attention_mask
        except Exception as e:
            self.logger.error(f"Error encoding SMILES: {str(e)}")
            raise


class MolecularDataProcessor:
    """
    Molecular Data Processor
    """
    def __init__(self, args):
        """
        Initializing the Data Processor
        Args:
            max_length: Maximum sequence length
            batch_size: Batch size
        """
        self.max_length = args.max_seq_length
        self.tokenizer = MolecularTokenizer(args)
        self.logger = logging.getLogger(__name__)

    def process_sequence_batch(self, batch_data):
        try:
            print(f"Processing batch of size {len(batch_data)}")
            input_ids, attention_mask = self.tokenizer.encode(batch_data, padding=True)
            print(f"Processed shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            self._validate_batch(input_ids, attention_mask)
            return input_ids, attention_mask
        except Exception as e:
            print(f"Error in process_sequence_batch: {str(e)}")
            raise

    def _validate_batch(self, input_ids, attention_mask):
        """Validating dimensions and values for batch data"""
        assert input_ids.size(1) == self.max_length, \
            f"Unexpected input_ids length: {input_ids.size(1)}, expected: {self.max_length}"
        assert attention_mask.size(1) == self.max_length, \
            f"Unexpected attention_mask length: {attention_mask.size(1)}, expected: {self.max_length}"
        assert torch.all((attention_mask == 0) | (attention_mask == 1)), \
            "Attention mask should only contain 0s and 1s"
