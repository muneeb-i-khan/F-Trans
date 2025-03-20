import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .bcm_layers import FastBlockCirculantLinear

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
        
        # Multi-head attention calculation
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
        # Use simple average pooling
        sequence_output = encoded.mean(dim=1)  # Average pooling over seq_len
        logits = self.classifier(sequence_output)
        return logits
