# -*- encoding: utf-8 -*-

# PIL (Pillow)：Python 最常用的图像处理库，类似 Android 的 Bitmap/BitmapFactory。
# 可以打开各种格式图片，并与 numpy 数组互转，供 OpenCV 等库使用。
from io import BytesIO
from pathlib import Path
from typing import Any, Union  # Union：联合类型，等价于 Kotlin 的联合类型或 TS 的 A | B | C

import cv2
import numpy as np  # NumPy ndarray：Python 中的多维数组，图像在内存中以 ndarray 形式存储
from PIL import Image, UnidentifiedImageError

root_dir = Path(__file__).resolve().parent
# Union[str, np.ndarray, bytes, Path, Image.Image] 等价于 TS: string | ndarray | bytes | Path | Image
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class LoadImage:
    """图像加载工具类，统一将多种输入格式转换为 BGR numpy 数组（OpenCV 标准格式）输出。"""

    def __init__(self):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        """
        __call__ 是 Python 魔术方法，定义后实例可像函数一样直接调用。
        等价于 Kotlin 的 operator fun invoke()。
        用法：loader = LoadImage(); result = loader(img)
        """
        # InputType.__args__ 获取 Union 中所有类型的元组，用于 isinstance 多类型检查
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        """根据输入类型加载图像为 numpy 数组（原始像素，未做颜色空间转换）。"""
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = self.img_to_ndarray(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            # BytesIO 将 bytes 包装成内存文件，让 PIL 可以像读文件一样读取，类似 Java 的 ByteArrayInputStream
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        """
        PIL Image → numpy 数组。
        PIL mode='1' 是 1 位黑白二值图，需先转为 8 位灰度图 'L'，再用 np.array() 转换。
        """
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type: Any) -> np.ndarray:
        """
        统一转换为 BGR 三通道格式（OpenCV 标准）。
        注意：PIL/文件/bytes 读出的图片颜色通道顺序是 RGB，OpenCV 要求 BGR，需要做转换。
        ndarray 输入默认已是 BGR，不做转换。
        """
        if img.ndim == 2:  # 灰度图（无通道维度）
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]  # shape = (H, W, C)，取第三维即通道数
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)  # 灰度 + Alpha

            if channel == 3:
                # PIL/文件/bytes 来源的图片是 RGB，需转为 BGR
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img  # 已是 ndarray（BGR），直接返回

            if channel == 4:
                return self.cvt_four_to_three(img)  # RGBA → BGR

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR：用 Alpha 做透明度混合，透明区域填白色。"""
        img_gray = img[..., 0]   # [..., 0] 取所有行列的第 0 个通道
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR：去掉 Alpha 通道，透明区域填白色。"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))  # PIL 来源是 RGBA，转为 OpenCV 的 BGR 顺序

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)

        mean_color = np.mean(new_img)  # np.mean()：计算数组所元素的平均值
        if mean_color <= 0.0:
            new_img = cv2.add(new_img, not_a)
        else:
            new_img = cv2.bitwise_not(new_img)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass