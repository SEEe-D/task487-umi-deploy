"""Optional directional opening offsets, applied once to original policy values."""
from dataclasses import dataclass

import numpy as np

MAX_COMPENSATION_DEG = 5.0
INTENT_REVERSAL_DEG = 0.5


@dataclass(frozen=True)
class GripperIntent:
    extreme_deg: float
    direction: int = 0  # -1 closing, +1 opening, 0 no direction established yet

    def advance(self, requested_deg):
        if self.direction < 0:
            if requested_deg >= self.extreme_deg + INTENT_REVERSAL_DEG:
                return GripperIntent(requested_deg, 1)
            return GripperIntent(min(self.extreme_deg, requested_deg), -1)
        if self.direction > 0:
            if requested_deg <= self.extreme_deg - INTENT_REVERSAL_DEG:
                return GripperIntent(requested_deg, -1)
            return GripperIntent(max(self.extreme_deg, requested_deg), 1)
        if requested_deg >= self.extreme_deg + INTENT_REVERSAL_DEG:
            return GripperIntent(requested_deg, 1)
        if requested_deg <= self.extreme_deg - INTENT_REVERSAL_DEG:
            return GripperIntent(requested_deg, -1)
        return self


def compensate_grippers(targets, intents, *, close_deg, open_deg, open_limits_deg):
    """Preview both hands; commit only first nonstale intent after acceptance.

    Direction comes from original policy requests, never offset targets or
    subsequently measured compression. Speculative horizon reversals do not
    advance cross-chunk state. The scheduler retimes this path before dispatch.
    """
    adjusted = targets.copy()
    next_intents = list(intents)
    for hand, column in enumerate((6, 13)):
        future = intents[hand]
        for i, requested in enumerate(targets[:, column]):
            future = future.advance(float(requested))
            if i == 0:
                next_intents[hand] = future
            offset = -close_deg if future.direction < 0 else open_deg if future.direction > 0 else 0.
            adjusted[i, column] = np.clip(requested + offset, 0., open_limits_deg[hand])
    return adjusted, tuple(next_intents)
