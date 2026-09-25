from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
import torch.nn as nn
import torch
import math

try:
    from .configuration_qraxai import GPTConfig
except ImportError:
    from configuration_qraxai import GPTConfig

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.q_proj(x)
        V = self.v_proj(x)
        K = self.k_proj(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            torch.finfo(scores.dtype).min
        )

        attention_w = torch.softmax(
            scores,
            dim=-1
        )

        output = attention_w @ V

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            seq_len,
            embed_dim
        )

        # Final projection
        output = self.o_proj(output)

        return output


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(embed_dim, 8)

        self.ln2 = nn.LayerNorm(embed_dim)

        # feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=4*embed_dim
            ),

            nn.GELU(),

            nn.Linear(
                in_features=4*embed_dim,
                out_features=embed_dim
            )
        )

    def forward(self, x):
        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x

class QraXAiForCausalLM(PreTrainedModel, GenerationMixin):

    config_class = GPTConfig

    def __init__(
        self,
        config
    ):

        super().__init__(config)

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.embed_dim
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(
            config.embed_dim
        )


        self.lm_head = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=False
        )

        self.post_init()

    def forward(
        self,
        input_ids,
        labels=None,
        **kwargs
    ):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) "
                f"cannot be greater than "
                f"max_seq_len ({self.config.max_seq_len})"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        token_emb = self.token_embedding(
            input_ids
        )

        pos_emb = self.position_embedding(
            positions
        )

        x = token_emb + pos_emb

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:

            shift_logits = logits[
                :, :-1, :
            ].contiguous()

            shift_labels = labels[
                :, 1:
            ].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1)
                ),
                shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )