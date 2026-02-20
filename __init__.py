from .directional_blur import (
    NODE_CLASS_MAPPINGS as _blur_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _blur_names,
)
from .average_masks import (
    NODE_CLASS_MAPPINGS as _avg_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _avg_names,
)

NODE_CLASS_MAPPINGS = {**_blur_classes, **_avg_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**_blur_names, **_avg_names}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
