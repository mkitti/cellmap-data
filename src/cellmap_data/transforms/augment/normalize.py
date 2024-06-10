from typing import Any, Dict
import torchvision.transforms.v2 as T


class Normalize(T.Transform):
    def __init__(self):
        super().__init__()

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        if x.max() - x.min() == 0:
            return x
        else:
            return (x - x.min()) / (x.max() - x.min())
