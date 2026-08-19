import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import apply_action_prefill


def test_action_prefill_clamps_prefix_and_leaves_generated_suffix_untouched():
    generated = jnp.arange(20 * 4, dtype=jnp.float32).reshape(1, 20, 4)
    known = jnp.full((1, 20, 4), -7.0, dtype=jnp.float32)
    mask = jnp.arange(20)[None, :] < 10

    result = np.asarray(apply_action_prefill(generated, known, mask))

    np.testing.assert_allclose(result[:, :10], -7.0)
    np.testing.assert_allclose(result[:, 10:], np.asarray(generated)[:, 10:])
