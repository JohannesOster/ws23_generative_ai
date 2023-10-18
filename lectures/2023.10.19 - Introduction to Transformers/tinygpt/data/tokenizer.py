import tiktoken


class Tokenizer():
    def __init__(self, unique_tokens=None):
        base_enc = tiktoken.get_encoding("gpt2")
        self.padding_token = '<pad>'
        self._padding_idx = base_enc.max_token_value+1

        self.enc = tiktoken.Encoding(
            name="gpt2",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={
                **base_enc._special_tokens,
                self.padding_token: self._padding_idx,
            }
        )

        self.n_vocab = self.enc.n_vocab
        self._unique_tokens = unique_tokens

        if unique_tokens:
            self.set_unique_tokens(unique_tokens)

    def set_unique_tokens(self, tokens):
        self._unique_tokens = sorted(list(tokens) + [self._padding_idx])
        self.n_vocab = len(self._unique_tokens)
        self.padding_idx = self._unique_tokens.index(self._padding_idx)

        # Precompute the encoding dictionaries for better performance then index lookup
        self._encode_dict = {token_id: idx for idx, token_id in enumerate(self._unique_tokens)}

    def encode(self, data):
        token_ids = self.enc.encode(data, allowed_special={'<pad>'})
        if self._unique_tokens:
            return [self._encode_dict[token_id] for token_id in token_ids]
        return token_ids

    def decode(self, data):
        if (self._unique_tokens):
            data = [self._unique_tokens[token_id] for token_id in data]
        return self.enc.decode(data)
