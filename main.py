from src.embedding.embedding import Embedding
from src.tokenizer.tokenizer import Tokenizer

embedding = Embedding()
tokenizer = Tokenizer()

user_msg = str(input("Digite algo: "))
tokens = tokenizer.tokenize(user_msg)
embeddings = embedding.generate_embedding(tokens)
print(embeddings)

