from .logger import get_logger
import os
import platform
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
# onnxruntime：加载并运行 ONNX 格式的 AI 模型（.onnx 文件）
# ONNX 是通用模型交换格式，PyTorch/TensorFlow 训练的模型可导出为 .onnx 后用此库高效推理
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,   # 核心类：加载 .onnx 模型并执行推理
    SessionOptions,
    get_available_providers,
    get_device,
)


class EP(Enum):
    """onnxruntime 推理时使用的硬件后端枚举（执行提供商 Execution Provider）。"""
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"    # NVIDIA GPU，需安装 onnxruntime-gpu
    DIRECTML_EP = "DmlExecutionProvider" # Windows DirectML GPU（支持 AMD/Intel GPU）


class OrtInferSession:
    """
    ONNX 模型推理会话封装，自动检测并选择最优执行后端（GPU 优先，兜底 CPU）。
    通过 __call__ 直接传入输入数据得到推理结果。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config 字典参数：
          - model_path: .onnx 模型文件路径
          - use_cuda: 是否启用 CUDA（NVIDIA GPU）
          - use_dml: 是否启用 DirectML（Windows GPU）
          - intra_op_num_threads / inter_op_num_threads: 推理线程数（可选）
        """
        self.logger = get_logger("OrtInferSession")

        model_path = config.get("model_path", None)
        self._verify_model(model_path)

        self.cfg_use_cuda = config.get("use_cuda", None)
        self.cfg_use_dml = config.get("use_dml", None)

        self.had_providers: List[str] = get_available_providers()
        EP_list = self._get_ep_list()  # 按优先级构建执行提供商列表，列表越靠前优先级越高

        sess_opt = self._init_sess_opts(config)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=EP_list,
        )
        self._verify_providers()

    @staticmethod
    def _init_sess_opts(config: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4           # 4=ERROR，只输出错误日志
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL  # 开启全部计算图优化

        cpu_nums = os.cpu_count()
        intra_op_num_threads = config.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = config.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def _get_ep_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """构建执行提供商列表，优先级：DirectML > CUDA > CPU。"""
        cpu_provider_opts = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(EP.CPU_EP.value, cpu_provider_opts)]

        cuda_provider_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        self.use_cuda = self._check_cuda()
        if self.use_cuda:
            EP_list.insert(0, (EP.CUDA_EP.value, cuda_provider_opts))

        self.use_directml = self._check_dml()
        if self.use_directml:
            self.logger.info(
                "Windows 10 or above detected, try to use DirectML as primary provider"
            )
            directml_options = (
                cuda_provider_opts if self.use_cuda else cpu_provider_opts
            )
            EP_list.insert(0, (EP.DIRECTML_EP.value, directml_options))
        return EP_list

    def _check_cuda(self) -> bool:
        if not self.cfg_use_cuda:
            return False

        cur_device = get_device()
        if cur_device == "GPU" and EP.CUDA_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.CUDA_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("!!!Recommend to use rapidocr_paddle for inference on GPU.")
        self.logger.info(
            "(For reference only) If you want to use GPU acceleration, you must do:"
        )
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`."
        )
        self.logger.info(
            "\tNote the onnxruntime-gpu version must match your cuda and cudnn version."
        )
        self.logger.info(
            "\tYou can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
            EP.CUDA_EP.value,
        )
        return False

    def _check_dml(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            self.logger.warning(
                "DirectML is only supported in Windows OS. The current OS is %s. Use %s inference by default.",
                cur_os,
                self.had_providers[0],
            )
            return False

        cur_window_version = int(platform.release().split(".")[0])
        if cur_window_version < 10:
            self.logger.warning(
                "DirectML is only supported in Windows 10 and above OS. The current Windows version is %s. Use %s inference by default.",
                cur_window_version,
                self.had_providers[0],
            )
            return False

        if EP.DIRECTML_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.DIRECTML_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("If you want to use DirectML acceleration, you must do:")
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
            EP.DIRECTML_EP.value,
        )
        return False

    def _verify_providers(self):
        session_providers = self.session.get_providers()
        first_provider = session_providers[0]

        if self.use_cuda and first_provider != EP.CUDA_EP.value:
            self.logger.warning(
                "%s is not avaiable for current env, the inference part is automatically shifted to be executed under %s.",
                EP.CUDA_EP.value,
                first_provider,
            )

        if self.use_directml and first_provider != EP.DIRECTML_EP.value:
            self.logger.warning(
                "%s is not available for current env, the inference part is automatically shifted to be executed under %s.",
                EP.DIRECTML_EP.value,
                first_provider,
            )

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        """
        执行模型推理，实例可像函数一样调用（等价于 Kotlin 的 operator fun invoke()）。
        将输入数据列表与模型输入节点名称配对后送入推理会话。
        """
        # dict(zip(names, data))：将两个列表对应配对成字典，类似 Kotlin 的 names.zip(data).toMap()
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)  # None 表示返回所有输出节点
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        # 列表推导式 [expr for item in iterable]，等价于 Kotlin 的 map { }
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        """从模型元数据中读取字符列表（用于 OCR 类模型）。"""
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class ONNXRuntimeError(Exception):
    pass