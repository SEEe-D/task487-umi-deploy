"""Patch orbax for CephFS: write directly to final path, no tmp dir, no rename."""
import logging

def _patch():
    from orbax.checkpoint._src.path import atomicity
    from etils import epath

    # 1. _create_tmp_directory: return final path directly (no tmp suffix)
    async def _direct_create(final_path, *args, **kwargs):
        import os
        p = str(final_path) if not isinstance(final_path, str) else final_path
        os.makedirs(p, exist_ok=True)
        return epath.Path(p)

    atomicity._create_tmp_directory = _direct_create

    # 2. Patch AtomicRenameTemporaryPath.finalize to skip rename (src == dst)
    _orig_finalize = atomicity.AtomicRenameTemporaryPath.finalize
    def _safe_finalize(self, *args, **kwargs):
        if str(self._tmp_path) == str(self._final_path):
            return  # no rename needed, already at final path
        try:
            _orig_finalize(self, *args, **kwargs)
        except Exception:
            pass  # ignore rename errors on CephFS

    atomicity.AtomicRenameTemporaryPath.finalize = _safe_finalize

    # 3. on_commit_callback: make safe
    _orig_commit = atomicity.on_commit_callback
    def _safe_commit(*args, **kwargs):
        try:
            _orig_commit(*args, **kwargs)
        except Exception:
            pass

    atomicity.on_commit_callback = _safe_commit

    logging.info("CephFS patch applied: direct write, no tmp dir, no rename")

_patch()
