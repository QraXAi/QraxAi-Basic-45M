from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):

    model_type = "qrax_ai"

    def __init__(
        self,
        vocab_size=10000,
        n_layers=6,
        max_seq_len=512,
        embed_dim=256,
        use_cache=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim