import os
from bindcraft import design_workers as dw

class Device:
    def __init__(self, i, kind): self.id, self.kind = i, kind
    def __str__(self): return f'{self.kind}:{self.id}'
    platform = property(lambda self: self.kind)
    def memory_stats(self): return {}

print("A) two devices on a platform that is neither cuda nor rocm")
dw.design_devices = lambda: [Device(0,'tpu'), Device(1,'tpu')]
try:
    print("   plan_design_workers ->", dw.plan_design_workers({}))
except Exception as e:
    print("   plan_design_workers raised %s: %s" % (type(e).__name__, e))

print()
print("B) cuda, process given a subset of the box: CUDA_VISIBLE_DEVICES=2,3")
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
dw.design_devices = lambda: [Device(0,'cuda'), Device(1,'cuda')]
print("   selected_design_gpus() ->", dw.selected_design_gpus())
for i, g in enumerate(dw.selected_design_gpus()):
    print(f"   worker {i} launched with {dw.design_visibility_variable()}={g}")
