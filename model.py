import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int  = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-4

    #for kv cache
    max_batch_size: int = 32 
    max_seq_len:int = 2048

    device: str = None

def precompute_theta_pos_embeddings(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2, "According to paper, must be even"
    # formula =>  theta_i = 10000^(-2(i-1)/dim), i = [1, 2, ... dim/2]
    #acc to paper, Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim,2).float()
    # shape = (head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the position (the "m" parameter)
    # shape = seq_len
    m = torch.arange(seq_len,  device=device)
    # multiply each theta by each position using the outer product
    # shape : (seq_len) outer product (head_dim) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex number in polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # shape: (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch_size, seq_len, H, dim) -> (batch_size, seq_len, H, dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = torch.unsqueeze(0).unsqueeze(2)
    # (batch_size, seq_len, H, dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (batch_size, seq_len, H, head_dim/2) -> (batch_size, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch_size, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else: 
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim= True) + self.eps)

    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, dim)

        # (B, 1, dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # applying RoPE (no change in shape of tensor)
        xq = apply_rotary_embeddings(xq, freq_complex, device = x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device = x.device)

        #replace the entry in cache for token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        #retrieve all the cached keys and values 
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # repeat the heads of K and V to reach the number of head of queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_kv) -> (B, H_Q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / torch.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)

        # (B, H_Q, 1, seq_len_kv) @ (B, H_Q, seq_len_kv, head_dim) -> (B, H_Q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, head_dim) -> (B, 1, H_Q, head_dim) -> (B, 1, Dim)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()    

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier)
        # round the hidden dim to nearest multiple of the multiple_of parameter
        hidden = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = x_v * swish  
        x = self.w2(x) 
        return x  
    
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()    
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention()
        self.feed_forward = FeedForward()

        #normalization before attention
        self.attention_norm = RMSNorm(args.dim, eps =args.norm_eps)
        #normalization before feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
      
    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(x))
        return out

class Transformers(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()    

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)  

        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_embeddings(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        #(B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        #(B, Seq_Len) -> (B, Seq_Len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) coressponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        #Consecutively apply to all layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output    
    


