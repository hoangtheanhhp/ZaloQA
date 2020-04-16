from QASystem.preprocess import ZaloDatasetProcessor
import kashgari
from kashgari.embeddings import BERTEmbedding

vocab_path = "../model/vocab.txt"
model = "../model/"
checkpoint_path = ""

# Tokenizer initialzation

embed = BERTEmbedding(model,
                      task=kashgari.CLASSIFICATION
                      , sequence_length=30)
tokenizer = embed.tokenizer
processor = ZaloDatasetProcessor()
processor.load_from_path("/home/anhht/Project/ZaloQA/Dataset", "train")
data = processor.train_data
tokenizer.tokenize(data[0].get('question') + data[0].get('text'))
