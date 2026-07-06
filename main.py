from src.embedding.embedding import Embedding
from src.tokenizer.tokenizer import Tokenizer
from src.self_attention.self_attention import SelfAttention

embedding = Embedding()
tokenizer = Tokenizer()
self_attention = SelfAttention()

user_msg = str(input("Digite algo: "))
tokens = tokenizer.tokenize(user_msg)
print(f"tokens: \n {tokens}")

embeddings = embedding.generate_embedding(tokens)
print(f"embeddings: \n {embeddings}")

attention = self_attention.forward(embeddings)
print(f"attention: \n {attention}")

