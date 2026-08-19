"""Patch orbax + etils for CephFS compatibility.

Run: uv run python src/openpi/training/patch_orbax.py
"""


def main():
    import importlib

    # 1. Patch etils epath backend.py — rmtree ignore_errors
    print("=== Patching etils epath backend.py ===")
    mod1 = importlib.import_module("etils.epath.backend")
    f1 = mod1.__file__
    with open(f1) as f:
        code1 = f.read()
    if "ignore_errors" not in code1 and "shutil.rmtree(path)" in code1:
        with open(f1 + ".bak", "w") as f:
            f.write(code1)
        code1 = code1.replace("shutil.rmtree(path)", "shutil.rmtree(path, ignore_errors=True)")
        with open(f1, "w") as f:
            f.write(code1)
        print(f"  Patched: {f1}")
    else:
        print(f"  Already patched or pattern not found: {f1}")

    # 2. Patch orbax atomicity.py — rename with copytree fallback
    print("=== Patching orbax atomicity.py ===")
    mod2 = importlib.import_module("orbax.checkpoint._src.path.atomicity")
    f2 = mod2.__file__
    with open(f2) as f:
        lines = f.readlines()

    if any("copytree" in l for l in lines):
        print(f"  Already patched: {f2}")
        return

    with open(f2 + ".bak", "w") as f:
        f.writelines(lines)

    new_lines = []
    for line in lines:
        if "self._tmp_path.rename(self._final_path)" in line and "try" not in line:
            indent = "    "  # 4 spaces, matching line 297
            new_lines.append(indent + "try:\n")
            new_lines.append(indent + "    self._tmp_path.rename(self._final_path)\n")
            new_lines.append(indent + "except Exception:\n")
            new_lines.append(indent + "    import shutil as _s\n")
            new_lines.append(indent + "    _s.copytree(str(self._tmp_path), str(self._final_path), dirs_exist_ok=True)\n")
            new_lines.append(indent + "    _s.rmtree(str(self._tmp_path), ignore_errors=True)\n")
            print("  Patched rename")
        else:
            new_lines.append(line)

    with open(f2, "w") as f:
        f.writelines(new_lines)
    print(f"  Saved: {f2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
