## setup 
```import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
```

## BCM Layer
```class BlockCirculantLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=32):
        super(BlockCirculantLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # Calculate number of blocks
        self.n_blocks_rows = math.ceil(out_features / block_size)
        self.n_blocks_cols = math.ceil(in_features / block_size)
        
        # Pad dimensions to be divisible by block_size
        self.in_features_padded = self.n_blocks_cols * block_size
        self.out_features_padded = self.n_blocks_rows * block_size
        
        # Instead of storing the full weight matrix, just store the first row of each block
        # This significantly reduces the parameter count
        self.weight_first_row = nn.Parameter(
            torch.Tensor(self.n_blocks_rows, self.n_blocks_cols, block_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_first_row, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_first_row)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def expand_to_circulant(self, first_row):
        """Expand first row to a full circulant matrix"""
        block_size = self.block_size
        circulant = torch.zeros(block_size, block_size, device=first_row.device)
        
        for i in range(block_size):
            circulant[i] = torch.roll(first_row, i, dims=0)
        
        return circulant
    
    def forward(self, input):
        # Pad input if necessary
        if input.size(-1) < self.in_features_padded:
            padding = self.in_features_padded - input.size(-1)
            input = F.pad(input, (0, padding))
        
        # Reshape input for block operations
        batch_size = input.size(0)
        input_reshaped = input.view(batch_size, self.n_blocks_cols, self.block_size)
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_features_padded, device=input.device)
        output_view = output.view(batch_size, self.n_blocks_rows, self.block_size)
        
        # Perform block circulant matrix multiplication
        for i in range(self.n_blocks_rows):
            for j in range(self.n_blocks_cols):
                first_row = self.weight_first_row[i, j]
                block = self.expand_to_circulant(first_row)
                output_view[:, i, :] += torch.bmm(
                    input_reshaped[:, j, :].unsqueeze(1),
                    block
                ).squeeze(1)
        
        # Reshape and trim output to desired dimensions
        output = output[:, :self.out_features]
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
            
        return output
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, block_size={self.block_size}'
```

## FFT
```class FastBlockCirculantLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=32):
        super(FastBlockCirculantLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # Calculate number of blocks
        self.n_blocks_rows = math.ceil(out_features / block_size)
        self.n_blocks_cols = math.ceil(in_features / block_size)
        
        # Pad dimensions to be divisible by block_size
        self.in_features_padded = self.n_blocks_cols * block_size
        self.out_features_padded = self.n_blocks_rows * block_size
        
        # Store only first row of each block
        self.weight_first_row = nn.Parameter(
            torch.Tensor(self.n_blocks_rows, self.n_blocks_cols, block_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_first_row, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_first_row)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # Pad input if necessary
        if input.size(-1) < self.in_features_padded:
            padding = self.in_features_padded - input.size(-1)
            input = F.pad(input, (0, padding))
        
        batch_size = input.size(0)
        input_reshaped = input.view(batch_size, self.n_blocks_cols, self.block_size)
        
        output = torch.zeros(batch_size, self.out_features_padded, device=input.device)
        output_view = output.view(batch_size, self.n_blocks_rows, self.block_size)
        
        # Process each block with FFT-based circulant multiplication
        for i in range(self.n_blocks_rows):
            for j in range(self.n_blocks_cols):
                x = input_reshaped[:, j, :]  # (batch_size, block_size)
                w = self.weight_first_row[i, j]  # (block_size)
                
                # Use FFT for fast circulant matrix multiplication
                # For circulant matrix C and vector x, C·x = IFFT(FFT(c) ⊙ FFT(x))
                # where c is the first column of C and ⊙ is element-wise multiplication
                x_fft = torch.fft.rfft(x, n=2*self.block_size)
                w_fft = torch.fft.rfft(w, n=2*self.block_size)
                output_fft = x_fft * w_fft.unsqueeze(0)
                result = torch.fft.irfft(output_fft, n=2*self.block_size)
                
                # Take only the first block_size elements
                output_view[:, i, :] += result[:, :self.block_size]
        
        # Trim output to desired dimensions
        output = output[:, :self.out_features]
        
        # Add bias
        if self.bias is not None:
            output += self.bias
            
        return output
```

## Shallow Transformer
```
class BCMTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=4, dim_feedforward=512, 
                 dropout=0.1, block_size=32, max_seq_length=512):
        super(BCMTransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        # Self-attention with BCM
        self.q_proj = FastBlockCirculantLinear(embed_dim, embed_dim, block_size)
        self.k_proj = FastBlockCirculantLinear(embed_dim, embed_dim, block_size)
        self.v_proj = FastBlockCirculantLinear(embed_dim, embed_dim, block_size)
        self.out_proj = FastBlockCirculantLinear(embed_dim, embed_dim, block_size)
        
        # Feed-forward with BCM
        self.linear1 = FastBlockCirculantLinear(embed_dim, dim_feedforward, block_size)
        self.linear2 = FastBlockCirculantLinear(dim_feedforward, embed_dim, block_size)
        
        # Other layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize positional encoding
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Self-attention
        residual = x
        batch_size, seq_len, embed_dim = x.size()
        
        # Reshape for BCM operations
        x_flat = x.reshape(-1, embed_dim)  # (batch_size * seq_len, embed_dim)
        
        # Compute Q, K, V projections
        q = self.q_proj(x_flat).view(batch_size, seq_len, embed_dim)
        k = self.k_proj(x_flat).view(batch_size, seq_len, embed_dim)
        v = self.v_proj(x_flat).view(batch_size, seq_len, embed_dim)
        
        # Multi-head attention calculation (simplified for clarity)
        head_dim = embed_dim // 4  # nhead=4
        q = q.view(batch_size, seq_len, 4, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, 4, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, 4, head_dim).transpose(1, 2)
        
        # Attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size * seq_len, embed_dim)
        
        # Output projection
        out = self.out_proj(out).view(batch_size, seq_len, embed_dim)
        
        # Add & norm
        out = self.norm1(residual + self.dropout(out))
        
        # Feed-forward
        residual = out
        out_flat = out.reshape(-1, embed_dim)
        
        ff_out = self.linear2(F.relu(self.linear1(out_flat))).view(batch_size, seq_len, embed_dim)
        
        # Add & norm
        out = self.norm2(residual + self.dropout(ff_out))
        
        return out

class BCMTransformerForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, num_classes, block_size=32, embed_dim=256):
        super(BCMTransformerForSequenceClassification, self).__init__()
        
        self.encoder = BCMTransformerEncoder(vocab_size, embed_dim=embed_dim, block_size=block_size)
        self.classifier = FastBlockCirculantLinear(embed_dim, num_classes, block_size)
        
    def forward(self, x, mask=None):
        encoded = self.encoder(x, mask)
        # Use CLS token or simple average pooling
        sequence_output = encoded.mean(dim=1)  # Average pooling over seq_len
        logits = self.classifier(sequence_output)
        return logits
```

## Dataset prep
```
def tokenize_and_prepare_imdb():
    # Set up tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
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
            text_list.append(processed_text)
        
        # Pad sequences
        max_length = max(len(text) for text in text_list)
        padded_texts = []
        attention_masks = []
        
        for text in text_list:
            padding = [vocab['<pad>']] * (max_length - len(text))
            padded_text = text + padding
            mask = [1] * len(text) + [0] * (max_length - len(text))
            
            padded_texts.append(padded_text)
            attention_masks.append(mask)
        
        return (
            torch.tensor(padded_texts, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
            torch.tensor(label_list, dtype=torch.long)
        )
    
    # Create data loaders
    train_iter, test_iter = IMDB()
    train_dataloader = DataLoader(
        list(train_iter), batch_size=16, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        list(test_iter), batch_size=16, shuffle=False, collate_fn=collate_batch
    )
    
    return train_dataloader, test_dataloader, vocab
```
## Train and eval
```def train_and_evaluate_model():
    # Get data
    train_dataloader, test_dataloader, vocab = tokenize_and_prepare_imdb()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model (choose appropriate block_size for your compression rate)
    model = BCMTransformerForSequenceClassification(
        vocab_size=len(vocab), 
        num_classes=2,
        block_size=32,
        embed_dim=256
    ).to(device)
    
    # Calculate parameter count
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Standard model for comparison (using regular Linear layers)
    # ...
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (texts, masks, labels) in enumerate(train_dataloader):
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, masks, labels in test_dataloader:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                outputs = model(texts, masks)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")
    
    return model
```
## Compression analysis
```
def analyze_compression(model):
    # Calculate parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    bcm_params = count_parameters(model)
    
    # Theoretical full-size model
    # Calculate how many parameters a standard model would have used
    standard_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, FastBlockCirculantLinear):
            # Full matrix would have in_features * out_features parameters
            full_params = module.in_features * module.out_features
            # BCM version has n_blocks * block_size parameters
            bcm_params = module.n_blocks_rows * module.n_blocks_cols * module.block_size
            standard_params += full_params
    
    # Calculate compression ratio
    compression_ratio = standard_params / bcm_params
    
    print(f"Standard model parameters (theoretical): {standard_params:,}")
    print(f"BCM model parameters: {bcm_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return compression_ratio
```
