"""Unsigned 128-bit products and bounded exact roots using signed-safe words."""

from __future__ import annotations

import torch

from .circuit_quotient import unsigned_product_limbs

MASK32 = 0xFFFFFFFF


def unsigned_product_words(a, b):
    """Return four little-endian uint32 words stored in int64 tensors.

    Inputs are unsigned 64-bit bit patterns in int64 storage, including
    negative signed representations. All intermediate arithmetic fits int64.
    """
    al, ah = a & MASK32, (a >> 32) & MASK32
    bl, bh = b & MASK32, (b >> 32) & MASK32
    h00, l00 = unsigned_product_limbs(al, bl)
    h01, l01 = unsigned_product_limbs(al, bh)
    h10, l10 = unsigned_product_limbs(ah, bl)
    h11, l11 = unsigned_product_limbs(ah, bh)
    middle = h00 + l01 + l10
    upper = h01 + h10 + l11 + (middle >> 32)
    return l00, middle & MASK32, upper & MASK32, h11 + (upper >> 32)


def unsigned_square_words(a):
    al, ah = a & MASK32, (a >> 32) & MASK32
    h00, l00 = unsigned_product_limbs(al, al)
    h01, l01 = unsigned_product_limbs(al, ah)
    h11, l11 = unsigned_product_limbs(ah, ah)
    middle = h00 + 2 * l01
    upper = 2 * h01 + l11 + (middle >> 32)
    return l00, middle & MASK32, upper & MASK32, h11 + (upper >> 32)


def subtract_words(a, b):
    """Unsigned modular subtraction; callers ensure nonnegative exact results."""
    borrow = 0
    result = []
    for x, y in zip(a, b, strict=True):
        value = x - y - borrow
        result.append(value & MASK32)
        borrow = (value < 0).long()
    return tuple(result)


def greater_words(a, b):
    greater = torch.zeros_like(a[0], dtype=torch.bool)
    equal = torch.ones_like(greater)
    for x, y in zip(reversed(a), reversed(b), strict=True):
        greater = greater | (equal & (x > y))
        equal = equal & (x == y)
    return greater


def words_to_float(a):
    value = a[3].double()
    for word in reversed(a[:3]):
        value = value * 4294967296.0 + word.double()
    return value


def floor_sqrt_bounded_words(a):
    """Exact floor sqrt for 0 <= value < 2**124.

    FP64 sqrt can be hundreds of integers away. Compute the exact signed
    residual via word subtraction, apply one rounded Newton correction,
    then compare exact squares on both sides. Roots and corrections fit int64.
    """
    y = words_to_float(a).sqrt().long()
    square = unsigned_square_words(y)
    negative = greater_words(square, a)
    larger = tuple(torch.where(negative, x, z) for x, z in zip(square, a, strict=True))
    smaller = tuple(torch.where(negative, z, x) for x, z in zip(square, a, strict=True))
    magnitude = words_to_float(subtract_words(larger, smaller))
    residual = torch.where(negative, -magnitude, magnitude)
    y = y + torch.round(residual / (2 * y.double()).clamp_min(1)).long()
    y = y - greater_words(unsigned_square_words(y), a).long()
    y = y + (~greater_words(unsigned_square_words(y + 1), a)).long()
    return y


def exact_product_match(pairs, target):
    if not 1 < target < 2**128:
        raise ValueError("Require a nontrivial target within 128 bits")
    words = unsigned_product_words(pairs[..., 0], pairs[..., 1])
    result = ((pairs != 0) & (pairs != 1)).all(-1)
    for i, word in enumerate(words):
        result = result & (word == ((target >> (32 * i)) & MASK32))
    return result
