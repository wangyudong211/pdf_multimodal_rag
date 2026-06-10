# AutoTokenizer：自动选择对应的分词器，将文本切分为 token ID 序列（NLP 模型的输入格式）
from transformers import AutoTokenizer, AutoModel
import torch

# nomic-ai/nomic-embed-text-v1.5
class TextEmbedding:
    """文本向量化模型封装，将字符串转换为 embedding 向量，用于向量数据库存储和相似度检索。"""

    def __init__(self, text_embedding_model_path):
        self.init_text_model(text_embedding_model_path)

    def init_text_model(self, text_embedding_model_path):
        print('init text embedding model...')
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(text_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.text_model = self.text_model.eval()

    def get_text_embedding(self, text):
        # padding=True：将同一 batch 中的序列填充到相同长度；truncation=True：超长文本截断
        text_inputs = self.text_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            text_outputs = self.text_model(**text_inputs)
        # mean(dim=1)：对所有 token 的向量取平均，得到整段文本的语义向量
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        return text_embeddings[0].detach().numpy()
        # .detach()：从计算图中分离张量（推理结束后不再需要梯度追踪）；.numpy()：转为 numpy 数组