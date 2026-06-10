import os
from PIL import Image  # PIL：Python 图像处理库，用于打开图片文件并传入图像 embedding 模型

class EmbeddingParse:
    """
    PDF 解析与向量化的主流程编排类。
    将 PDF 文本和图片分别通过对应模型转换为 embedding 向量，供后续存入向量数据库。
    """

    def __init__(self, pdf_extract, text_split, text_embedding, image_embedding):
        self.pdf_extract = pdf_extract
        self.text_split = text_split
        self.text_embedding = text_embedding
        self.image_embedding = image_embedding

    def parse(self, pdf_path, output_dir):
        """
        完整解析流程：PDF → 提取文本/图片/表格 → 文本分块 → 分别向量化。
        返回：(texts, texts_embedding, images_embedding)
          - texts: LangChain Document 对象列表，包含文本内容和元数据
          - texts_embedding: 每个文本块对应的 numpy 向量列表
          - images_embedding: 每张图片对应的 numpy 向量列表
        """
        print('start parse pdf...')
        texts, consp_time = self.pdf_extract.extract_pdf(pdf_path, output_dir)
        texts = self.text_split.split_texts(texts)
        texts = self.text_split.add_meta_data(texts, pdf_path)

        if len(texts) != 0:
            # 列表推导式：对每个文本块调用 get_text_embedding，text.page_content 是 LangChain Document 的文本字段
            texts_embedding = [self.text_embedding.get_text_embedding(text.page_content) for text in texts]

        images_embedding = []
        image_files = os.listdir(output_dir)
        for img in image_files:
            try:
                image = Image.open(os.path.join(output_dir, img))
                image_embedding = self.image_embedding.get_image_embedding(image)
                images_embedding.append(image_embedding)
            except Exception as e:
                print(f'Error processing: {img}: {e}')

        return texts, texts_embedding, images_embedding