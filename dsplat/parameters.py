"""Constraints of per-atom parameters: free components, step scales and feasible sets.

An atom's parameters are tensors with one row per atom, for example surfel
centers [S, 3]. Every solver step works in dimensionless coordinates and goes
through ParameterConfig: scale sets the unit, project restores feasibility, and
frozen components keep their values.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ParameterConfig:
    trainable: bool | tuple[bool, ...] = True  # one bool, or one per component
    scale: float = 1.0  # physical length of a unit optimizer step
    lower: float | tuple[float, ...] | None = None
    upper: float | tuple[float, ...] | None = None
    period: tuple[float, ...] | None = None
    normalize: bool = False  # unit vectors, such as surfel normals

    def mask(self, value):
        mask = torch.as_tensor(self.trainable, dtype=torch.bool, device=value.device)
        return mask.expand(value.shape[1] if value.ndim == 2 else 1)

    def project(self, proposed, reference):
        """Feasible values closest to proposed; frozen components come from reference."""
        scalar = proposed.ndim == 1
        value = proposed[:, None] if scalar else proposed
        if self.period is not None:
            value = value.remainder(value.new_tensor(self.period))
        if self.lower is not None:
            value = torch.maximum(value, value.new_tensor(self.lower))
        if self.upper is not None:
            value = torch.minimum(value, value.new_tensor(self.upper))
        frozen = reference[:, None] if scalar else reference
        mask = self.mask(reference)
        value = torch.where(mask[None], value, frozen)
        if self.normalize and mask.any():
            # The free components take whatever length the frozen ones leave to 1.
            fixed = torch.where(mask[None], 0, value)
            free = value - fixed
            # A free part of zero length has no direction; it takes the reference's.
            free = torch.where(
                free.norm(dim=-1, keepdim=True) > 0, free, torch.where(mask[None], frozen, 0)
            )
            room = (1 - fixed.square().sum(-1, keepdim=True)).clamp(min=0).sqrt()
            length = free.norm(dim=-1, keepdim=True)
            if bool(((room > 0) & (length == 0)).any()):
                raise ValueError("Cannot normalize a zero vector without a reference direction.")
            value = fixed + free * room / length.clamp(min=1e-30)
        return value[:, 0] if scalar else value

    def tangent(self, vector, value):
        """The part of an update [..., S, D] that project keeps at a unit vector value [S, D].

        Frozen components drop out, and so does the radial part within the free
        components, whose squared length is 1 minus that of the frozen ones.
        """
        if not self.normalize:
            return vector
        mask = self.mask(value)
        free = torch.where(mask, value, 0)
        room = (1 - torch.where(mask, 0, value).square().sum(-1, keepdim=True)).clamp(min=0)
        vector = torch.where(mask, vector, 0)
        tangent = vector - (vector * free).sum(-1, keepdim=True) * free / room.clamp(min=1e-30)
        return torch.where(room > 0, tangent, 0)

    def differential(self, jacobian, value):
        """Chain a dictionary derivative [..., S, D] through the retraction, mask and scale."""
        if self.normalize:
            jacobian = self.tangent(jacobian, F.normalize(value, dim=-1))
        return jacobian[..., self.mask(value)] * self.scale


def periodic_delta(delta, period):
    """Differences folded into [-period/2, period/2] per axis; None leaves them unchanged."""
    if period is None:
        return delta
    period = torch.as_tensor(period, dtype=delta.dtype, device=delta.device)
    return delta - period * torch.round(delta / period)
