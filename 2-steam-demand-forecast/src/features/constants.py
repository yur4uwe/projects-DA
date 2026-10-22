"""Shared constants for promotional elasticity, storefront ramps, and post-sale gameplay tails."""

# Share of the full promotional exposure realised on sale day 0, 1, 2+ (24h-avg CCU integration)
SALE_RAMP: dict[int, float] = {0: 0.35, 1: 0.85}

# Post-sale gameplay installation and play tail decay parameters:
# Tail(Δt) = TAIL_INITIAL_FACTOR * exp(-Δt / TAIL_DECAY_RATE)
TAIL_INITIAL_FACTOR: float = 0.50
TAIL_DECAY_RATE: float = 2.5

# Standard economic price elasticity exponent for indie games:
# lift = (1.0 - discount_depth)^(-DEFAULT_ELASTICITY_EPSILON) - 1.0
DEFAULT_ELASTICITY_EPSILON: float = 0.8
WEEKEND_BOOST: float = 0.15
