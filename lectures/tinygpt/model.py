from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    n_embd = 14596      # = vocab_size = number of distinct tokens
    d_embd = 512        # = dimension of one embedding vector
    n_head = 8          # = number of attention heads in the Multihead Attention Layer
    d_ff = 1024         # = dimension of the feed-forward layer (4*d_embd)
    n_layers = 4        # = number of transformer block layers
    n_block = 256         # = block size, the max amount of tokens processable by LLM
    dropout = 0.1       # = probability of dropout in the dropout layers
    batch_size = 12     # = how many sequences are procsses simultaneously

    padding_idx = 14595


class TinyGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.w_embd = nn.Embedding(config.n_embd, config.d_embd, padding_idx=config.padding_idx)
        self.pos_embd = nn.Embedding(config.n_block, config.d_embd)
        self.dropout = nn.Dropout(p=config.dropout)
        self.t_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.linear = nn.Linear(config.d_embd, config.n_embd, bias=False)

        self.register_buffer('pos', torch.arange(0, config.n_block, dtype=torch.long).unsqueeze(0))

        # Tie the weights of the final layer to the word embedding layer
        # https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/model.py#L171
        self.linear.weight = self.w_embd.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # These values are not mentioned in any of the papers but the
        # gpt2 reference implementation did the same and it showed significant
        # improvement in the initial loss (300 vs 9 initial loss)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, key_padding_mask=None):
        batch_size, _ = x.size()
        pos = self.pos.repeat(batch_size, 1)

        x = self.w_embd(x) + self.pos_embd(pos)
        x = self.dropout(x)
        for transformer_block in self.t_blocks:
            x = transformer_block(x, key_padding_mask)
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer('attn_mask', torch.triu(torch.ones(
            config.n_block, config.n_block), diagonal=1).bool())

        self.ln_1 = nn.LayerNorm(config.d_embd)
        self.att = nn.MultiheadAttention(config.d_embd, config.n_head,
                                         dropout=config.dropout, bias=False, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(config.d_embd, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embd),
        )
        self.ln_2 = nn.LayerNorm(config.d_embd)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.att(x, x, x, attn_mask=self.attn_mask,
                                  key_padding_mask=key_padding_mask, average_attn_weights=False)
        x = self.ln_1(x + attn_output)
        x = self.ln_2(x + self.ff(x))
        return x
