from typing import Callable, Mapping, Optional, Sequence
import torch
import tensorstore
from upath import UPath
from .source import CellMapSource, EmptySource


class CellMapArray:
    def __init__(
        self,
        array_info: Mapping[str, Sequence[int | float]],
        class_paths: Mapping[str, str],
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        value_transforms: Optional[
            Callable | Sequence[Callable] | Mapping[str, Callable]
        ] = None,
        empty_value: float | int = torch.nan,
        pad: bool = False,
        interpolation: str = "nearest",
        context: Optional[tensorstore.Context] = None,
        axis_order: str = "zyx",
    ):
        self.array_info = array_info
        self.classes = list(class_paths.keys())
        self.class_paths = class_paths
        self.class_relation_dict = class_relation_dict
        self.value_transforms = value_transforms
        self.empty_value = empty_value
        self.pad = pad
        self.interpolation = interpolation
        self.context = context
        self.axis_order = axis_order
        self.has_data = False
        self.empty_store = torch.ones(self.array_info["shape"]) * self.empty_value  # type: ignore
        self.sources = {}
        for i, (class_name, class_path) in enumerate(self.class_paths.items()):
            if UPath(class_path).exists():
                if isinstance(self.value_transforms, dict):
                    value_transform: Callable = self.value_transforms[class_name]
                elif isinstance(self.value_transforms, list):
                    value_transform: Callable = self.value_transforms[i]
                else:
                    value_transform: Callable = self.value_transforms  # type: ignore
                source = CellMapSource(
                    path=class_path,
                    target_class=class_name,
                    target_scale=self.array_info["scale"],
                    target_voxel_shape=self.array_info["shape"],  # type: ignore
                    pad=self.pad,
                    pad_value=self.empty_value,
                    interpolation=self.interpolation,
                    axis_order=self.axis_order,
                    value_transform=value_transform,
                    context=self.context,
                )
                if not self.has_data:
                    self.has_data = source.class_counts != 0
            else:
                if (
                    self.class_relation_dict is not None
                    and class_name in self.class_relation_dict
                ):
                    # Add lookup of source images for true-negatives in absence of annotations
                    source = self.class_relation_dict[class_name]
                else:
                    source = EmptySource(
                        class_name,
                        self.array_info["scale"],
                        self.array_info["shape"],  # type: ignore
                        self.empty_store,
                    )
            self.sources[class_name] = source
            # Check to make sure we aren't trying to define true negatives with non-existent images
            for class_name in self.classes:
                if isinstance(self.sources[class_name], (CellMapSource, EmptySource)):
                    continue
                is_empty = True
                for other_label in self.sources[class_name]:
                    if other_label in self.sources and isinstance(
                        self.sources[other_label], CellMapSource
                    ):
                        is_empty = False
                        break
                if is_empty:
                    self.sources[class_name] = EmptySource(
                        class_name, self.array_info["scale"], self.array_info["shape"], empty_store  # type: ignore
                    )

    def set_spatial_transforms(self, spatial_transforms):
        self._current_spatial_transforms = spatial_transforms
        for source in self.sources.values():
            source.set_spatial_transforms(spatial_transforms)
