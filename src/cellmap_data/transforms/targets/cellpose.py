from cellpose.dynamics import masks_to_flows_gpu_3d, masks_to_flows
from cellpose.dynamics import masks_to_flows_gpu as masks_to_flows_gpu_2d
import torch


class CellposeFlow:
    """
    Cellpose flow transform.

    Args:
        ndim (int): Number of dimensions.
        device (str | None, optional): Device to use. Defaults to None
            (use GPU if available, else CPU).
    """

    def __init__(self, ndim: int, device: str | None = None):
        UserWarning("This is still in development and may not work as expected")
        self.ndim = ndim
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        _device = torch.device(device)
        if device == "cuda" or device == "mps":
            if ndim == 3:
                flows_func = lambda x: masks_to_flows_gpu_3d(x, device=_device)
            elif ndim == 2:
                flows_func = lambda x: masks_to_flows_gpu_2d(x, device=_device)
            else:
                raise ValueError(f"Unsupported dimension {ndim}")
        else:
            flows_func = lambda x: masks_to_flows(x, device=_device)
        self.flows_func = flows_func
        self.device = _device

    def __call__(self, masks):
        flows, centers = self.flows_func((masks > 0).squeeze().numpy())
        flows = flows[None, ...]
        if self.ndim == 2:
            flows = flows[None, ...]
        return torch.tensor(flows)
