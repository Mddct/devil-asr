import torch


class LFR(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor,
                inputs_mask: torch.Tensor) -> torch.Tensor:
        pass
