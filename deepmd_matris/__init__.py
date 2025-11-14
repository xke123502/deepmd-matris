"""DeePMD-MatRIS: MatRIS integration for DeePMD-kit."""

from .argcheck import matris_model_args
# 导入 model_e0 以确保 @BaseModel.register("matris") 被执行
from . import model_e0  # noqa: F401

__version__ = "0.1.0"
__all__ = ["matris_model_args"]