#coding: utf-8

# fitz：PyMuPDF 库的模块名，用于解析 PDF 文件（提取文本、图片、渲染页面为图像）
import fitz
import os
import time
import cv2
import magic  # python-magic：通过文件内容（魔数）检测真实的 MIME 类型，而非依赖文件扩展名
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

from src.table_det import TableDetector
from src.table_det.utils.extract_img import extract_table_img

class PDFExtract:
    """PDF 内容提取器，负责从 PDF 中提取文本、嵌入图片和表格（截图保存）。"""

    def __init__(self, table_det_model):
        self.table_det_model = table_det_model

    def extract_images_from_pdf(self, pdf_path, output_dir):
        """提取 PDF 中直接嵌入的图片（如插图、logo），保存为独立图片文件。"""
        pdf_doc = fitz.open(pdf_path)
        for p_num in range(len(pdf_doc)):
            page = pdf_doc[p_num]
            imgs = page.get_images(full=True)  # full=True：返回完整图片信息（含 xref 引用 ID）
            for img_idx, img in enumerate(imgs):
                xref = img[0]  # xref：PDF 内部图像资源的引用编号
                base_img = pdf_doc.extract_image(xref)
                img_bytes = base_img['image']
                img_ext = base_img['ext']

                img_filename = f"page_{p_num+1}_image_{img_idx+1}.{img_ext}"
                img_path = os.path.join(output_dir, img_filename)

                with open(img_path, 'wb') as img_file:
                    img_file.write(img_bytes)
                print('Saved: {}'.format(img_path))
        pdf_doc.close()

    def extract_text_from_pdf(self, pdf_path):
        """逐页提取 PDF 文本，返回每页文本字符串组成的列表。"""
        pdf_doc = fitz.open(pdf_path)
        texts = []
        for page in pdf_doc:
            text = page.get_text()  # get_text()：提取当前页的纯文本内容
            texts.append(text)
        return texts

    def extract_table_from_pdf(self, pdf_path, output_dir):
        """
        将 PDF 每页渲染为图像，用表格检测模型检测表格区域，
        再通过透视变换裁剪出表格图片并保存。
        """
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            # get_pixmap：将 PDF 页面渲染为位图，dpi=120 控制分辨率，alpha=False 不含透明通道
            pix = page.get_pixmap(dpi=120, alpha=False)
            # Image.frombytes：将原始字节数据按指定模式和尺寸构建 PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = np.array(img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PIL 输出 RGB，转为 OpenCV 的 BGR
            result = self.table_det_model(img_bgr)
            for table_num, res in enumerate(result):
                file_name = f"page_{page_num + 1}_table_{table_num + 1}.png"
                img_path = os.path.join(output_dir, file_name)
                lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
                wrapped_img = extract_table_img(img_bgr, lt, rt, rb, lb)
                cv2.imwrite(img_path, wrapped_img)
                print('Saved: {}'.format(img_path))
        pdf_doc.close()

    def extract_pdf(self, pdf_path, output_dir):
        """主入口：依次提取嵌入图片、文本、表格截图，返回文本列表和耗时。"""
        if not os.path.exists(pdf_path) and self.is_pdf(pdf_path):
            raise ValueError("Not find file path or not pdf file : {}".format(
                pdf_path))
        else:
            st_time = time.time()
            # image
            self.extract_images_from_pdf(pdf_path, output_dir)
            # text
            texts = self.extract_text_from_pdf(pdf_path)
            # table
            self.extract_table_from_pdf(pdf_path, output_dir)
            consp_time = time.time() - st_time
            print("Time taken to extract text and images: {} seconds".format(consp_time))
            return texts, consp_time

    def is_pdf(self, file_path):
        """通过文件内容的 MIME 类型判断是否为 PDF，避免被伪造的扩展名欺骗。"""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type == 'application/pdf'