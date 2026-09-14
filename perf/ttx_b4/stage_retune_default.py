"""Pin the harness baseline, then install the bit-exact retune as the shipped table.

Both edits in one script so there is no window in which a queued run can start against a
half-changed tree: the `on` arm must keep reading TODAY's blocks after the default moves, and a
snapshot of the live table (`dict(T._MM_BLOCK)`) stops being that the moment the default changes.
"""
import io, ast

# ---- 1. harness: the `on` arm's baseline becomes a literal, not a snapshot -------------------
p = "perf/other512/fold_ab_multi.py"
s = io.open(p).read()
old = "    _MM_BLOCK_SHIPPED = dict(T._MM_BLOCK)\n"
assert s.count(old) == 1
new = '''    # The `on` arm's block table is a LITERAL, not `dict(T._MM_BLOCK)`. It used to be the
    # snapshot, and that is safe only while the default never moves: the moment the retune below
    # becomes the shipped table, a snapshot makes `on` read the retuned blocks and the A/B reports
    # ~1.000x for a lever that is really worth 1.0375x. These are the values `main` shipped before
    # the retune landed, so the arm keeps measuring against the same baseline it always did.
    _MM_BLOCK_SHIPPED = {
        (8, 24): (4, 8, 1, 4, 1), (8, 8): (4, 8, 1, 4, 1),
        (4, 12): (4, 4, 1, 4, 1), (4, 4): (4, 4, 1, 4, 1),
        (4, 16): (4, 4, 1, 4, 1), (4, 17): (4, 4, 1, 4, 1),
        (2, 12): (4, 2, 1, 4, 1), (2, 2): (4, 2, 1, 4, 1),
        (12, 36): (4, 12, 1, 2, 1), (12, 12): (8, 12, 1, 2, 1),
    }
    assert set(_MM_BLOCK_SHIPPED) == set(T._MM_BLOCK), (
        "the live table gained or lost a key; the pinned baseline has to gain it too")
'''
s = s.replace(old, new)
ast.parse(s)
harness = s

# ---- 2. engine: the retune becomes the default ----------------------------------------------
q = "tt_bio/tenstorrent.py"
t = io.open(q).read()
RETUNE = {
    (8, 24): ((4, 8, 2, 2, 2), "1.4473x"),
    (4, 4):  ((8, 4, 1, 4, 1), "1.2561x"),
    (2, 2):  ((8, 2, 1, 4, 1), "1.3345x"),
    (2, 12): ((4, 2, 2, 2, 2), "1.1581x"),
    (4, 16): ((4, 4, 2, 2, 2), "1.0891x"),
    (4, 12): ((8, 4, 2, 2, 2), "1.0641x"),
    (4, 17): ((8, 4, 1, 4, 1), "1.0369x"),
}
OLD = {
    (8, 24): "(4, 8, 1, 4, 1)", (4, 4): "(4, 4, 1, 4, 1)", (2, 2): "(4, 2, 1, 4, 1)",
    (2, 12): "(4, 2, 1, 4, 1)", (4, 16): "(4, 4, 1, 4, 1)", (4, 12): "(4, 4, 1, 4, 1)",
    (4, 17): "(4, 4, 1, 4, 1)",
}
for key, (blk, ratio) in RETUNE.items():
    line_key = "    (%d, %d): %s," % (key[0], key[1], OLD[key])
    assert t.count(line_key) == 1, (line_key, t.count(line_key))
    t = t.replace(line_key, "    (%d, %d): %s," % (key[0], key[1], str(blk).replace(",)", ")")))
ast.parse(t)

io.open(p, "w").write(harness)
io.open(q, "w").write(t)
print("harness baseline pinned + 7 keys retuned")
