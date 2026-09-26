#!/usr/bin/env python3
"""The leaky control: storage groups hold their members STRONGLY again, patched at RUNTIME.

bcx-armtree's `27e000ff2`, on main as `a257dfbbc`, made `Tensor.shares` hold weak references so
a view and its source stop owning each other. That commit is inside the range between the tree
bcx-large measured the n=352 ceiling on (`68b7a49a0`) and main today, and main today shows no
per-block growth at all where bcx-large measured 0.0638 GB. Twenty commits touched `autograd.py`
in that range, so the flat line is only CORRELATED with the fix until something moves it back.

This inverts that one change and nothing else.

Patched rather than edited. bcx-armtree applied its control as a diff to the file; this row
cannot, because its two ladder arms share one worktree and a dirty `autograd.py` would silently
become the tree the next rung measures. `_share` and `_members` are referenced only inside
`tt_bio/autograd.py` and only by module-global name (`:456`, `:776`, `:912`, `:913`), so
rebinding the two module attributes reaches every caller and nothing else in the process.

Apply BEFORE the first taped call. Groups built under the weak implementation hold `weakref.ref`
objects, and the strong `_members` would hand those back as if they were tensors.
"""


def apply(autograd) -> dict:
    """Rebind `_share` and `_members` on the live module. Returns what it replaced."""
    old = {"_share": autograd._share, "_members": autograd._members}

    def _members(group) -> list:
        """The live tensors of a storage group. Strong members under the leaky control."""
        return list(group)

    def _share(a, b) -> None:
        group = a.shares if a.shares is not None else [a]
        for t in (b.shares if b.shares is not None else [b]):
            if not any(t is m for m in group):
                group.append(t)
        for t in group:
            t.shares = group

    autograd._members = _members
    autograd._share = _share
    return old


def verify(autograd) -> dict:
    """Prove the patch is live rather than trusting that it was called.

    The discriminator is what the GROUP LIST holds, not what `_members` returns. Under the weak
    implementation `_members` dereferences, so it hands back the tensors either way and a check
    written against it reads True on both arms -- a control gated like its subject, which tests
    nothing. `a.shares[0]` is a `weakref.ref` under main and the tensor itself under the control,
    and that is the thing that moves.
    """
    class _T:
        shares = None

    a, b = _T(), _T()
    autograd._share(a, b)
    held = type(a.shares[0]).__name__
    return {"group_len": len(a.shares), "group_holds": held,
            "strong": held != "ReferenceType",
            "members_len": len(autograd._members(a.shares))}
