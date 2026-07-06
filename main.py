from src.embedding.embedding import Embedding
from src.tokenizer.tokenizer import Tokenizer
from src.self_attention.self_attention import SelfAttention
from src.MLP.MLP import MLP

tokenizer = Tokenizer()
embedding = Embedding()
self_attention = SelfAttention()
mlp = MLP()

user_msg = str(input("Digite algo: "))
tokens = tokenizer.tokenize(user_msg)
print(f"tokens: \n {tokens} \n")

embeddings = embedding.generate_embedding(tokens)
print(f"embeddings: \n {embeddings} \n")

attention = self_attention.forward(embeddings)
print(f"attention: \n {attention} \n")

MLP_block = mlp.mlp_block(attention)
print(f"resultado MLP: \n {MLP_block} \n")

