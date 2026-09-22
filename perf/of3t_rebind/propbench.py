import timeit

class Slotted:
    __slots__ = ("value",)
    def __init__(self, v): self.value = v

class Prop:
    __slots__ = ("_value",)
    def __init__(self, v): self._value = v
    @property
    def value(self): return self._value
    @value.setter
    def value(self, v): self._value = v

N = 5_000_000
for cls in (Slotted, Prop):
    o = cls(object())
    t = timeit.timeit("o.value", globals={"o": o}, number=N)
    print("%-8s read  %.1f ns" % (cls.__name__, t / N * 1e9))
