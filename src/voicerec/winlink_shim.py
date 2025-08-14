# src/voicerec/winlink_shim.py
def patch_links():
    import os, shutil
    from pathlib import Path

    def _copy2(src, dst):
        # force to plain str
        src = os.fspath(src)
        dst = os.fspath(dst)

        # ensure parent dir exists
        parent = os.path.dirname(dst) or "."
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

        # skip if same file
        try:
            if os.path.exists(src) and os.path.exists(dst):
                try:
                    if os.path.samefile(src, dst):
                        return
                except Exception:
                    pass
        except Exception:
            pass

        # robust copy
        try:
            shutil.copy2(src, dst)
        except PermissionError:
            shutil.copy(src, dst)
        except TypeError:
            # raw stream copy fallback
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

    # ---- os.* patches ----
    def _safe_link(src, dst, *a, **k): _copy2(src, dst)
    def _safe_symlink(src, dst, *a, **k): _copy2(src, dst)

    os.link = _safe_link
    os.symlink = _safe_symlink

    # ---- pathlib.Path.* patches (this is what SB calls) ----
    _orig_symlink_to = getattr(Path, "symlink_to", None)
    if _orig_symlink_to:
        def _path_symlink_to(self, target, *a, **k):
            # destination is 'self', source is 'target'
            _copy2(target, self)
        Path.symlink_to = _path_symlink_to

    # Python 3.10 has Path.hardlink_to
    _orig_hardlink_to = getattr(Path, "hardlink_to", None)
    if _orig_hardlink_to:
        def _path_hardlink_to(self, target, *a, **k):
            _copy2(target, self)
        Path.hardlink_to = _path_hardlink_to

    # Environment knobs for HF Hub (keeps cache local & avoids link tricks)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_HARD_LINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
