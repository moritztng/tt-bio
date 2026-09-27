"""Put the taped-softmax lever IN or OUT, anchored to `_softmax_fw_config` only.

`return kwargs.get("compute_kernel_config") or precise_config()` also appears in the taped
matmul, where it is the BACKWARD config and has nothing to do with this A/B. A plain sed patches
both and the base arm then measures two levers.
"""
import sys

P = "tt_bio/taped_ttnn.py"
LEVER = '    return kwargs.get("compute_kernel_config") or precise_config()'
BASE = '    return kwargs.get("compute_kernel_config")  # BASE ARM: no precise default'
want = sys.argv[1]

src = open(P).read()
i = src.index("def _softmax_fw_config(kwargs):")
j = src.index("\n@_verb(", i)
head, body, tail = src[:i], src[i:j], src[j:]
assert body.count(LEVER) + body.count(BASE) == 1, "the helper's return line is not unique"
body = body.replace(BASE, LEVER) if want == "fix" else body.replace(LEVER, BASE)
open(P, "w").write(head + body + tail)
print("arm=%s lever_in=%s" % (want, LEVER.strip() in body))
