from typing import Any
import torch

class STEArgmax(torch.autograd.Function):
    """A torch autograd function defining the argmax used in MixPrec"""

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x: torch.Tensor = args[0]
        return torch.nn.functional.one_hot(
                torch.argmax(x, dim = 0),
                num_classes = len(x)).t().to(dtype = torch.float32)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        return grad_outputs[0], None
