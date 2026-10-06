"""Read the three EFFECT arms and print what separates them.

The arms are the same seed and the same 24-step loop; one has no added term, one adds
`aromatic_binder` at weight 1.0 and one at 5.0. The quantity reported is the binder's mean W/F/Y
probability, recomputed in NumPy from the design logits rather than read off the objective.
"""
import json
import statistics
import sys

WINDOW = 8


def row(path):
    data = json.load(open(path))
    trace = data["trace"]
    tail = [step["binder_aromatic"] for step in trace[-WINDOW:]]
    return {"first": trace[0]["binder_aromatic"], "last": trace[-1]["binder_aromatic"],
            "mean": statistics.mean(tail), "sd": statistics.pstdev(tail),
            "final_loss": data["final_loss"], "terms": data["terms"],
            "steps": data["steps"]}


def main() -> None:
    arms = {name: row(path) for name, path in (pair.split("=", 1) for pair in sys.argv[1:])}
    print("{:10s} {:>10s} {:>10s} {:>12s} {:>10s} {:>11s}".format(
        "arm", "step 1", "last", f"mean last {WINDOW}", "sd", "final loss"))
    for name, arm in arms.items():
        print("{:10s} {:10.6f} {:10.6f} {:12.6f} {:10.6f} {:11.6f}".format(
            name, arm["first"], arm["last"], arm["mean"], arm["sd"], arm["final_loss"]))
    base = next(iter(arms.values()))
    drift = abs(base["last"] - base["first"])
    print()
    for name, arm in arms.items():
        moved = arm["last"] - arm["first"]
        ratio = f"{moved / drift:.0f}x the control's own drift" if drift and name != "control" else ""
        print(f"{name}: moved {moved:+.6f} over {arm['steps']} steps {ratio}")
    print()
    names = sorted({key for arm in arms.values() for key in arm["terms"]})
    print("{:22s}".format("term") + "".join(f"{name:>12s}" for name in arms))
    for term in names:
        print("{:22s}".format(term) + "".join(
            f"{arm['terms'].get(term, float('nan')):12.4f}" for arm in arms.values()))


if __name__ == "__main__":
    main()
