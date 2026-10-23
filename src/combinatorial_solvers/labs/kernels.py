"""Single-site exact differences and FFT all-site LABS differences."""

from __future__ import annotations

import torch


def rounded_integer(values):
    rounded = values.round()
    error = (values - rounded).abs().amax()
    if not bool(torch.isfinite(values).all()) or error.item() >= 0.125:
        raise ArithmeticError("FFT output is not sufficiently close to integer values")
    return rounded.to(torch.int64), float(error.item())


def fft_quantities(spins, *, gains=False):
    size = spins.shape[-1]
    length = 1 << (2 * size - 2).bit_length()
    spectrum = torch.fft.rfft(spins.to(torch.float64), n=length)
    power = spectrum.real.square() + spectrum.imag.square()
    full, error = rounded_integer(torch.fft.irfft(power, n=length))
    corr = full[:, 1:size]
    energy = corr.square().sum(-1)
    if not gains:
        return corr, energy, None, error
    field, field_error = rounded_integer(
        torch.fft.irfft(spectrum * (power - size), n=length)[:, :size]
    )
    mirror, mirror_error = rounded_integer(
        torch.fft.irfft(spectrum * spectrum, n=length)[:, : 2 * size : 2]
    )
    delta = -4 * spins.to(torch.int64) * field + 4 * (size - 2 + mirror)
    return corr, energy, delta, max(error, field_error, mirror_error)


def site_difference(spins, corr, indices):
    size = spins.shape[-1]
    lag = torch.arange(1, size, device=spins.device)
    before, after = indices[:, None] - lag, indices[:, None] + lag
    left = spins.gather(1, before.clamp_min(0)) * (before >= 0)
    right = spins.gather(1, after.clamp_max(size - 1)) * (after < size)
    selected = spins.gather(1, indices[:, None])
    change = -2 * selected.to(torch.int64) * (left + right).to(torch.int64)
    delta = (2 * corr * change + change.square()).sum(-1)
    return change, delta


def apply_flip(spins, corr, energy, indices, change, delta, accepted):
    corr.add_(change * accepted[:, None])
    energy.add_(delta * accepted)
    old = spins.gather(1, indices[:, None])
    spins.scatter_(1, indices[:, None], torch.where(accepted[:, None], -old, old))


def choose_tabu(delta, energy, best, tabu_until, step, uniforms):
    allowed = (tabu_until < step) | (energy[:, None] + delta < best[:, None])
    scores = delta.masked_fill(~allowed, torch.iinfo(torch.int64).max)
    lowest = scores.min(-1).values
    tied = allowed & (scores == lowest[:, None])
    # IID random keys give uniform choice among equal integer scores.
    indices = uniforms.masked_fill(~tied, -1).argmax(-1)
    return indices, tied.any(-1)
