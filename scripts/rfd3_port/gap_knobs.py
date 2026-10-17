"""The RFD3 knob table the gap_* scripts share: arm token -> env flag the model reads at import."""
import os

ENV_KNOBS = {"gsm": "RFD3_GATHERED_SOFTMAX", "bs": "RFD3_BLOCK_SPARSE",
             "pzc": "RFD3_PROCESS_Z_COLLAPSE", "fgrid": "RFD3_FAST_GRID",
             "asdpa": "RFD3_ATOM_SDPA", "tb8": "RFD3_TRANSITION_B8",
             "dsdpa": "RFD3_DIT_SDPA", "hifi2": "RFD3_HIFI2", "sf": "RFD3_SIGMOID_FOLD",
             "rs": "RFD3_RESIDENT_STEP", "pws": "RFD3_PAIR_WS", "kd": "RFD3_ATTN_KDTREE",
             "pwb8": "RFD3_PAIR_WS_B8", "at": "RFD3_ATTN_THREAD", "dpb": "RFD3_DPB",
             "apb": "RFD3_ATOM_PB_FUSED", "pdpb": "RFD3_PF_DPB", "css": "RFD3_CSS",
             "pwr": "RFD3_PAIR_WS_RESID"}


def apply(arm):
    """Set every knob: on if named in the '+'-joined ARM, off otherwise. Call before tt_bio.rfd3 loads."""
    on = set(arm.split("+")) if arm else set()
    for k, env in ENV_KNOBS.items():
        os.environ[env] = "1" if k in on else "0"
    return on
