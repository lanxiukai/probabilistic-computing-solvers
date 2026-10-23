"""Explicit execution choices; proposal probabilities and RNG layouts stay fixed."""

PROFILES = {
    "reference_v1": (False, False, False),
    "buffer_v1": (True, False, False),
    "buffer_sqrt_v1": (True, True, False),
    "buffer_qr_v1": (True, False, True),
    "buffer_sqrt_qr_v1": (True, True, True),
}


def configure(sampler, profile):
    if profile not in PROFILES:
        raise ValueError("Unknown execution profile")
    buffer, sqrt, qr = PROFILES[profile]
    sampler.use_triton_sqrt = sqrt
    sampler.use_triton_pre_matches = qr
    sampler.metadata["execution_profile"] = profile
    return buffer
