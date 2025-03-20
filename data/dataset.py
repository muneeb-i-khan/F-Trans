import torch
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def tokenize_and_prepare_imdb(batch_size=16, max_seq_length=256):
    # Set up tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    # Create training iterator for vocabulary building
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # Text processing pipeline
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: 1 if x == 'pos' else 0
    
    def collate_batch(batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = text_pipeline(_text)
            # Truncate if too long
            if len(processed_text) > max_seq_length:
                processed_text = processed_text[:max_seq_length]
            text_list.append(processed_text)
        
        # Pad sequences
        max_batch_length = min(max(len(text) for text in text_list), max_seq_length)
        padded_texts = []
        attention_masks = []
        
        for text in text_list:
            padding = [vocab['<pad>']] * (max_batch_length - len(text))
            mask = [1] * len(text) + [0] * (max_batch_length - len(text))
            padded_text = text + padding
            
            padded_texts.append(padded_text)
            attention_masks.append(mask)
        
        return (
            torch.tensor(padded_texts, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
            torch.tensor(label_list, dtype=torch.long)
        )
    
    # Create fresh iterators for train and test
    train_iter, test_iter = IMDB()
    
    # Convert to list to allow multiple iterations over the dataset
    train_data = list(train_iter)
    test_data = list(test_iter)
    
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    
    return train_dataloader, test_dataloader, vocab
