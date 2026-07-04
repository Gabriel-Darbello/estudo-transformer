import numpy as np
from pathlib import Path
import json

class Embedding:
    def __init__(self):
        self.vocab_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "vocabulary.json"
        )

        self.embedding_dim = 3

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)


    def _get_embedding(self, token: str):
        if token not in self.vocabulary:
            self.vocabulary[token] = np.random.uniform(
                -1,
                1,
                self.embedding_dim
            ).tolist()

            with open(self.vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocabulary, f, indent=4, ensure_ascii=False)

        return self.vocabulary[token]

    def generate_embedding(self, tokens:list[str]) -> np.ndarray:
        return np.array(
            [self._get_embedding(token) for token in tokens]
        )