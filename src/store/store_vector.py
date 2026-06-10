import os

# qdrant_client：Qdrant 向量数据库的 Python 客户端
# Qdrant 是专门为 AI 应用设计的向量数据库，支持高效的近似最近邻（ANN）向量相似度搜索
from qdrant_client import QdrantClient, models
import numpy as np
import uuid

class StoreVec:
    """向量数据库管理类，负责创建集合（Collection）和存储文本/图片的 embedding 向量。"""

    def __init__(self, embedding_vec_path):
        print('init qdrant client...')
        os.makedirs(embedding_vec_path, exist_ok=True)
        # QdrantClient(path=...)：数据持久化到本地磁盘
        # 也可使用 QdrantClient(':memory:') 仅于存（不持久化）
        self.q_client = QdrantClient(path=embedding_vec_path)

    def create_client_collection(self, text_embedding_size, image_embedding_size):
        """
        创建 Qdrant 集合（类似数据库的表），分别存储文本向量和图片向量。
        集合已存在则跳过，避免重复创建。
        - size：向量维度，必须与 embedding 模型输出维度一致
        - distance=COSINE：使用余弦相似度衡量向量距离（RAG 场景的常用选择）
        """
        print('create qdrant client...')
        if not self.q_client.collection_exists(('texts')):
            self.q_client.create_collection(
                collection_name='texts',
                vectors_config=models.VectorParams(
                    size=text_embedding_size,
                    distance=models.Distance.COSINE
                )
            )

        if not self.q_client.collection_exists('images'):
            self.q_client.create_collection(
                collection_name='images',
                vectors_config=models.VectorParams(
                    size=image_embedding_size,
                    distance=models.Distance.COSINE
                )
            )

    def store_image_text_embedding(self, texts, texts_embedding, output_dir, image_files, images_embedding):
        """将文本和图片的 embedding 向量批量写入对应集合。"""
        print('store embedding to db...')
        # PointStruct：Qdrant 的数据点结构，包含 id（唯一标识）、vector（向量）、payload（附加元数据）
        # 列表推导式生成 PointStruct 列表，批量上传效率高于逐条插入
        self.q_client.upload_points(
            collection_name='texts',
            points=[
                models.PointStruct(
                    id=text.metadata['uuid'],
                    vector=np.array(texts_embedding[idx]),
                    payload={
                        'metadata': text.metadata,
                        'content': text.page_content  # LangChain Document 的文本字段
                    }
                )
                for idx, text in enumerate(texts)
            ]
        )

        if len(images_embedding) > 0:
            self.q_client.upload_points(
                collection_name='images',
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),  # 图片没有预置 uuid，用 uuid4() 动态生成
                        vector=np.array(images_embedding[idx]),
                        payload={
                            'image_path': output_dir + '/' + str(image_files[idx])
                        }
                    )
                    for idx in range(len(image_files))
                ]
            )