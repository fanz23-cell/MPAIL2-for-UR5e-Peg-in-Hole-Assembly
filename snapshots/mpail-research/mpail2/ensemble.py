import torch
from torch import nn
from typing import List
from .layers import resolve_nn_activation


class EnsembleLinear(nn.Module):
    """Linear layer with independent parameters for each ensemble member."""

    def __init__(
        self,
        ensemble_size: int,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.weight_v = nn.Parameter(torch.empty(ensemble_size, out_dim, in_dim))
            self.weight_g = nn.Parameter(torch.empty(ensemble_size, out_dim))
            self.weight: nn.Parameter | None = None
        else:
            self.weight = nn.Parameter(torch.empty(ensemble_size, out_dim, in_dim))
            self.weight_v = None
            self.weight_g = None
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_weight_norm:
            assert self.weight_v is not None and self.weight_g is not None
            for member_idx in range(self.ensemble_size):
                nn.init.kaiming_uniform_(self.weight_v[member_idx], a=5**0.5)
            with torch.no_grad():
                self.weight_g.copy_(self.weight_v.norm(dim=-1))
        else:
            assert self.weight is not None
            for member_idx in range(self.ensemble_size):
                nn.init.kaiming_uniform_(self.weight[member_idx], a=5**0.5)
        if self.bias is not None:
            bound = self.in_dim ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [E, ..., in_dim]
        if self.use_weight_norm:
            assert self.weight_v is not None and self.weight_g is not None
            norm = self.weight_v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            w = self.weight_g.unsqueeze(-1) * (self.weight_v / norm)
        else:
            assert self.weight is not None
            w = self.weight
        out = torch.einsum("e...i,eoi->e...o", x, w)
        if self.bias is None:
            return out
        bias_shape = (self.ensemble_size,) + (1,) * (x.ndim - 2) + (self.out_dim,)
        return out + self.bias.view(*bias_shape)

    def extra_repr(self) -> str:
        return (
            f"ensemble_size={self.ensemble_size}, "
            f"in_features={self.in_dim}, out_features={self.out_dim}, "
            f"bias={self.bias is not None}, use_weight_norm={self.use_weight_norm}"
        )


class EnsembleLayerNorm(nn.Module):
    """LayerNorm with independent affine params per ensemble member."""

    def __init__(self, ensemble_size: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ensemble_size, dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [E, ..., dim]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        affine_shape = (self.ensemble_size,) + (1,) * (x.ndim - 2) + (self.dim,)
        return x_norm * self.weight.view(*affine_shape) + self.bias.view(*affine_shape)

    def extra_repr(self) -> str:
        return (
            f"ensemble_size={self.ensemble_size}, "
            f"normalized_shape={self.dim}, eps={self.eps}, "
            f"elementwise_affine=True"
        )


class EnsembleMLP(nn.Module):
    """MLP with independent parameters, evaluated in parallel over ensemble axis."""

    def __init__(
        self,
        *,
        ensemble_size: int,
        input_dim: int,
        hidden_dims: List[int],
        activation: str,
        output_dim: int,
        use_layer_norm: bool = False,
        use_weight_norm: bool = False,
        disable_output_bias: bool = False,
        override_last_layer_norm: bool = False,
        override_last_layer_activation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation
        dims = [input_dim] + self.hidden_dims + [output_dim]

        layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_last = i == (len(dims) - 2)
            linear = EnsembleLinear(
                ensemble_size=ensemble_size,
                in_dim=dims[i],
                out_dim=dims[i + 1],
                bias=not (is_last and disable_output_bias),
                use_weight_norm=use_weight_norm,
            )
            layers.append(linear)
            should_ln = (not is_last and use_layer_norm) or (is_last and override_last_layer_norm)
            if should_ln:
                layers.append(EnsembleLayerNorm(ensemble_size=ensemble_size, dim=dims[i + 1]))
            if (not is_last) or override_last_layer_activation:
                layers.append(resolve_nn_activation(activation))  # Activation marker

        self.model = nn.Sequential(*layers)

    def extra_repr(self) -> str:
        return (
            f"ensemble_size={self.ensemble_size}, "
            f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, activation={self.activation!r}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[0] != self.ensemble_size:
            raise ValueError(
                f"Expected ensemble-first tensor with shape [E, ..., D], got {tuple(x.shape)} "
                f"for ensemble_size={self.ensemble_size}."
            )
        return self.model(x)
