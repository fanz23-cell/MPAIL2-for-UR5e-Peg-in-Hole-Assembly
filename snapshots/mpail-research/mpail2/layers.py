import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Tuple


class WeightNormLinear(nn.Module):
    """Linear layer with Salimans--Kingma weight norm (no nn.utils.weight_norm; deepcopy-safe)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.weight_g = nn.Parameter(torch.empty(out_features, 1, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_g.copy_(self.weight_v.norm(dim=1, keepdim=True).clamp(min=1e-12))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.weight_v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        w = self.weight_g * self.weight_v / norm
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class WeightNormConv2d(nn.Module):
    """Conv2d with weight norm over each output channel (matches nn.utils.weight_norm default)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, ...],
        stride: int | Tuple[int, ...] = 1,
        padding: int | Tuple[int, ...] = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size  # type: ignore[misc]
        if isinstance(stride, int):
            self.stride: Tuple[int, int] = (stride, stride)
        else:
            self.stride = stride  # type: ignore[assignment]
        if isinstance(padding, int):
            self.padding: Tuple[int, int] = (padding, padding)
        else:
            self.padding = padding  # type: ignore[assignment]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.weight_v = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw, **factory_kwargs)
        )
        self.weight_g = nn.Parameter(torch.empty(out_channels, 1, 1, 1, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            gn = torch.linalg.vector_norm(self.weight_v, dim=(1, 2, 3), keepdim=True).clamp(
                min=1e-12
            )
            self.weight_g.copy_(gn)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(self.weight_v, dim=(1, 2, 3), keepdim=True).clamp(
            min=1e-12
        )
        w = self.weight_g * self.weight_v / norm
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"
        )


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    elif act_name == "mish":
        return torch.nn.Mish()
    elif act_name == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def mlp_factory(
    *,
    input_dim: int,
    hidden_dims: List[int],
    activation: Callable,
    output_dim: int,
    use_layer_norm: bool = False,
    use_weight_norm: bool = False,
    disable_output_bias: bool = False,
    override_last_layer_norm: bool = False,
    override_last_layer_activation: bool = False,
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.Sequential:
    activation = resolve_nn_activation(activation)
    layers = []
    Lin = WeightNormLinear if use_weight_norm else nn.Linear

    def add_layer(*, layers, layer, use_layer_norm, layer_dim=None):
        layers.append(layer)
        if use_layer_norm and layer_dim is not None:
            layers.append(nn.LayerNorm(layer_dim))

    # Input layer
    add_layer(
        layers=layers, layer=Lin(input_dim, hidden_dims[0]),
        use_layer_norm=use_layer_norm, layer_dim=hidden_dims[0]
    )
    layers.append(activation)

    # Hidden layers
    for layer_index in range(len(hidden_dims)):
        is_last = layer_index == len(hidden_dims) - 1
        out_dim = output_dim if is_last else hidden_dims[layer_index + 1]
        # Don't apply normalization to output layer unless override_last_layer_norm is True
        add_layer(
            layers=layers,
            layer=Lin(
                hidden_dims[layer_index], out_dim,
                bias=not (is_last and disable_output_bias)
            ),
            use_layer_norm=(not is_last and use_layer_norm) or (is_last and override_last_layer_norm),
            layer_dim=out_dim
        )
        if not is_last:
            layers.append(activation)

    if override_last_layer_activation:
        layers.append(activation)

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model


def cnn_factory(
    *,
    H: int,
    W: int,
    C: int,
    output_dim: int,
    out_channels: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    activation: str = "silu",
    hidden_dims: List[int] = [],
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    use_weight_norm: bool = False,
    override_last_layer_norm: bool = False,
    override_last_layer_activation: bool = False,
    **kwargs
) -> torch.nn.Sequential:

    num_layers = len(kernel_sizes)
    assert num_layers == len(strides), "kernel_sizes and strides must have the same length"

    layers = []
    Lin = WeightNormLinear if use_weight_norm else nn.Linear
    Conv = WeightNormConv2d if use_weight_norm else nn.Conv2d

    # Convolutional layers
    in_channels = C
    for i in range(num_layers):
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        padding = 0

        layers.append(Conv(in_channels, out_channels[i], kernel_size, stride, padding))

        layers.append(resolve_nn_activation(activation))

        in_channels = out_channels[i]

    layers.append(nn.Flatten())

    # Compute the size of the flattened feature map
    with torch.no_grad():
        dummy_input = torch.zeros(1, C, H, W)
        dummy_output = nn.Sequential(*layers)(dummy_input)
        flattened_size = dummy_output.shape[1]

    in_dim = flattened_size
    for i, dim in enumerate(hidden_dims):
        layers.append(Lin(in_dim, dim))
        layers.append(resolve_nn_activation(activation))
        in_dim = dim

    # Final linear layer to project to output_dim
    layers.append(Lin(in_dim, output_dim))

    if override_last_layer_norm:
        layers.append(nn.LayerNorm(output_dim))

    if override_last_layer_activation:
        layers.append(resolve_nn_activation(activation))

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model


def identity_factory(
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.Identity:
    model = nn.Identity()
    model.to(device=device, dtype=dtype)
    return model