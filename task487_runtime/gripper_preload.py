"""Optional position preload for a closing gripper, not contact/force feedback."""
from dataclasses import dataclass

import numpy as np


MAX_PRELOAD_DEG = 2.0  # Same travel as the installed driver's extra-close setting.
INTENT_REVERSAL_DEG = 0.5


@dataclass(frozen=True)
class ClosureIntent:
    extreme_deg: float
    closing: bool = False

    def advance(self, requested_deg):
        # Compare original policy requests, never the preloaded command or
        # measured compression: otherwise the preload can release itself.
        if self.closing:
            if requested_deg >= self.extreme_deg + INTENT_REVERSAL_DEG:
                return ClosureIntent(requested_deg, False)
            return ClosureIntent(min(self.extreme_deg, requested_deg), True)
        if requested_deg <= self.extreme_deg - INTENT_REVERSAL_DEG:
            return ClosureIntent(requested_deg, True)
        return ClosureIntent(max(self.extreme_deg, requested_deg), False)


def preload_right_gripper(targets, intent, preload_deg):
    """Preview a future path and return its next near-term request intent.

    Only the first non-stale request advances cross-chunk intent. Speculative
    closing/opening in the rest of the horizon must not commit that state.
    Caller commits this state only after accepting the entire scheduled path.
    The shared scheduler must retime these adjusted targets before dispatch.
    """
    adjusted = targets.copy()
    future, next_intent = intent, intent
    for i, requested in enumerate(targets[:, 6]):
        future = future.advance(float(requested))
        if i == 0:
            next_intent = future
        if future.closing:
            adjusted[i, 6] = max(0., requested - preload_deg)
    return adjusted, next_intent
