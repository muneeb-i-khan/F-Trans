import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastBlockCirculantLinear(nn.Module):
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
