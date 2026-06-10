import cv2
import numpy as np

from .det import Det
from .utils.load_image import LoadImage


class TableDetector:
    """
    表格检测器顶层封装，对外提供统一的调用接口。
    内部使用 Det 完成 ONNX 模型推理，输出每个表格的四角坐标。
    """

    def __init__(
        self,
        use_cuda=False,
        use_dml=False,
        model_path=None,
    ):
        self.img_loader = LoadImage()
        obj_det_config = {
            "model_path": model_path,
            "use_cuda": use_cuda,
            "use_dml": use_dml,
        }

        self.obj_detector = Det(obj_det_config)

    def __call__(
        self,
        img,
        det_accuracy=0.8,
    ):
        """
        检测图像中的表格，返回每个表格的边框坐标和四角点。
        返回列表中每个元素格式：{"box": [xmin,ymin,xmax,ymax], "lt": [x,y], "rt": [x,y], "rb": [x,y], "lb": [x,y]}
        """
        img = self.img_loader(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:-1]
        obj_det_res, pred_label = self.init_default_output(h, w)
        result = []
        obj_det_res, obj_det_elapse = self.obj_detector(img, score=det_accuracy)
        for i in range(len(obj_det_res)):
            det_res = obj_det_res[i]
            score, box = det_res
            xmin, ymin, xmax, ymax = box
            lb, lt, rb, rt = self.get_box_points(box)
            lb1, lt1, rb1, rt1 = self.get_real_rotated_points(
                lb, lt, pred_label, rb, rt
            )
            result.append(
                {
                    "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "lb": [int(lb1[0]), int(lb1[1])],
                    "lt": [int(lt1[0]), int(lt1[1])],
                    "rt": [int(rt1[0]), int(rt1[1])],
                    "rb": [int(rb1[0]), int(rb1[1])],
                }
            )
        return result

    def init_default_output(self, h, w):
        img_box = np.array([0, 0, w, h])
        # 初始化默认值
        obj_det_res, edge_box, pred_label = (
            [[1.0, img_box]],
            img_box.reshape([-1, 2]),
            0,
        )
        return obj_det_res, pred_label

    def add_pre_info_for_cls(self, cls_img, edge_box, xmin_cls, ymin_cls):
        """
        Args:
            cls_img:
            edge_box:
            xmin_cls:
            ymin_cls:

        Returns: 带边缘划线的图片，给方向分类提供先验信息

        """
        cls_box = edge_box.copy()
        cls_box[:, 0] = cls_box[:, 0] - xmin_cls
        cls_box[:, 1] = cls_box[:, 1] - ymin_cls
        # 画框增加先验信息，辅助方向label识别
        cv2.polylines(
            cls_img,
            [np.array(cls_box).astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(255, 0, 255),
            thickness=5,
        )

    def adjust_edge_points_axis(self, edge_box, lb, lt, rb, rt, xmin_edge, ymin_edge):
        edge_box[:, 0] += xmin_edge
        edge_box[:, 1] += ymin_edge
        lt, lb, rt, rb = (
            lt + [xmin_edge, ymin_edge],
            lb + [xmin_edge, ymin_edge],
            rt + [xmin_edge, ymin_edge],
            rb + [xmin_edge, ymin_edge],
        )
        return lb, lt, rb, rt

    def get_box_points(self, img_box):
        x1, y1, x2, y2 = img_box
        lt = np.array([x1, y1])  # 左上角
        rt = np.array([x2, y1])  # 右上角
        rb = np.array([x2, y2])  # 右下角
        lb = np.array([x1, y2])  # 左下角
        return lb, lt, rb, rt

    def get_real_rotated_points(self, lb, lt, pred_label, rb, rt):
        """根据方向分类结果（pred_label 0~3）旋转四角坐标，使 lt 始终对应实际左上角。"""
        if pred_label == 0:
            lt1 = lt
            rt1 = rt
            rb1 = rb
            lb1 = lb
        elif pred_label == 1:
            lt1 = rt
            rt1 = rb
            rb1 = lb
            lb1 = lt
        elif pred_label == 2:
            lt1 = rb
            rt1 = lb
            rb1 = lt
            lb1 = rt
        elif pred_label == 3:
            lt1 = lb
            rt1 = lt
            rb1 = rt
            lb1 = rb
        else:
            lt1 = lt
            rt1 = rt
            rb1 = rb
            lb1 = lb
        return lb1, lt1, rb1, rt1

    def pad_box_points(self, h, w, xmax, xmin, ymax, ymin, pad):
        ymin_edge = max(ymin - pad, 0)
        xmin_edge = max(xmin - pad, 0)
        ymax_edge = min(ymax + pad, h)
        xmax_edge = min(xmax + pad, w)
        return xmin_edge, ymin_edge, xmax_edge, ymax_edge