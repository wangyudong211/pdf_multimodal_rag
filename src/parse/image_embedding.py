# transformers：Hugging Face 出品的 NLP/多模态模型库，提供 BERT、GPT、CLIP 等预训练模型的统一接口
# AutoModel / AutoProcessor：根据模型目录中的配置文件自动选择对应的模型/预处理器类（无需写死具体类名）
from transformers import AutoModel, AutoProcessor
import torch  # PyTorch：深度学习框架，管理模型权重和张量计算

# nomic-ai/nomic-embed-vision-v1.5
class ImageEmbedding:
    """图像向量化模型封装，将 PIL Image 转换为固定维度的 embedding 向量，用于向量数据库存储和检索。"""

    def __init__(self, image_embedding_model_path):
        self.init_image_model(image_embedding_model_path)

    def init_image_model(self, image_embedding_model_path):
        print('init image embedding model...')
        # local_files_only=True：只从本地加载模型，不联网下载
        # trust_remote_code=True：允许执行模型目录中的自定义代码（某些模型需要）
        self.image_model = AutoModel.from_pretrained(image_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.image_processor = AutoProcessor.from_pretrained(image_embedding_model_path, use_fast=True, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.image_model.eval()  # eval()：切换到推理模式，关闭 Dropout/BatchNorm 的训练行为

    def get_image_embedding(self, image):
        # image_processor 负责将 PIL Image 缩放、归一化并转换为模型输入格式
        # return_tensors='pt'：返回 PyTorch tensor（pt=PyTorch，tf=TensorFlow）
        image_inputs = self.image_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            # torch.no_grad()：推理时禁用梯度计算，节省内存和计算量（类似 Android 推理模式）
            image_outputs = self.image_model(**image_inputs)
            # **image_inputs：将字典解包为关键字参数传入，等价于 Kotlin 的 spread 操作
        # last_hidden_state：模型最后一层所有 token 的隐藏状态，shape = (batch, seq_len, dim)
        # .mean(dim=1)：对 seq_len 维度取平均，得到整图的单一向量表示
        image_embeddings = image_outputs.last_hidden_state
        return image_embeddings.mean(dim=1).squeeze().cpu().numpy()
        # .squeeze()：去除大小为1的多余维度；.cpu()：从 GPU 搬回 CPU；.numpy()：转为 numpy 数组