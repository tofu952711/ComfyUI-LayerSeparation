"""ComfyUI 详情图层分离自定义节点入口。

标准 ComfyUI V1 插件: 暴露 NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS。
中文翻译在 locales/zh/ 下, 由 ComfyUI 服务端自动扫描, 无需在此注册。
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
