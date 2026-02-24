from .directional_blur import (
    NODE_CLASS_MAPPINGS as _blur_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _blur_names,
)
from .average_masks import (
    NODE_CLASS_MAPPINGS as _avg_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _avg_names,
)
from .smooth_mask import (
    NODE_CLASS_MAPPINGS as _smooth_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _smooth_names,
)

NODE_CLASS_MAPPINGS = {**_blur_classes, **_avg_classes, **_smooth_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**_blur_names, **_avg_names, **_smooth_names}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
