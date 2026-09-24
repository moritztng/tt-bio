"""Record what the predictor actually sees: BC2 pads only the DESIGN chain."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc2_state as B
from bindcraft.af2 import pad_design_chains, protein_state_shapes, padded_prediction_length

s = B.campaign_settings()
ds, states, losses = B.design_state(s)
padded = pad_design_chains(states, 32, 0)
shapes = protein_state_shapes(padded)
n_raw = sum(sum(c.values()) for c in B.state_shape(states).values())
n_pad = sum(sum(c.values()) for c in B.state_shape(padded).values())
blob = {"raw": B.state_shape(states), "padded": B.state_shape(padded),
        "shapes": [[a, list(b), list(c)] for a, b, c in shapes],
        "n_raw": n_raw, "n_padded": n_pad,
        "n_device_bucket_32": padded_prediction_length(n_pad, 32),
        "stage_plan": B.stage_plan(s), "losses": sorted(losses)}
(HERE / "state_shape.json").write_text(json.dumps(blob, indent=1))
print(json.dumps(blob, indent=1))
