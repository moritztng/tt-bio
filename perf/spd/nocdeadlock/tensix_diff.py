#!/usr/bin/env python3
"""
Usage:
    tensix_diff

Description:
    Tensix-level state of a hung chip, stuck core against healthy cores. A core is stuck when one of its CBs has a
    pack-side local tiles_received that differs from the stream register the unpacker polls (the H5 signature). For
    every stuck core and the first two healthy ones it prints all three threads' GPRs and every Tensix debug-bus
    signal, then the signals whose value on a stuck core differs from both healthy cores. That separates the two
    readings of H5: the pack thread still holds STALLWAIT(THCON, PACK) + STOREREG because the packer never finished
    (pack request FIFO not empty, packer busy), or it issued the store and the store was lost (FIFO empty, GPR
    NUM_MSGS_RECEIVED already at the next count). Run it with hangcap.sh's environment and triage arguments.

Owner:
    spd-nocdeadlock2 (tt-bio)
"""
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from dispatcher_data import run as get_dispatcher_data
from elfs_cache import run as get_elfs_cache
from run_checks import run as get_run_checks
from triage import ScriptConfig, run_script
from dump_cb_counters import NUM_CBS, field, local_cbs, stream_reg

script_config = ScriptConfig(depends=["run_checks", "dispatcher_data", "elfs_cache"])

state = {"stuck": [], "healthy": []}


def classify(location, risc_name, dispatcher_data, elfs_cache, context):
    if str(risc_name) != "trisc2":
        return
    try:
        pack = local_cbs(location, "trisc2", dispatcher_data, elfs_cache)
    except Exception as e:
        print(f"{location.to_user_str()} cb_interface unreadable: {e!r}", flush=True)
        return
    short = []
    for cb in range(NUM_CBS):
        if not field(pack[cb], "fifo_num_pages"):
            continue
        local = (field(pack[cb], "tiles_acked_received_init") or 0) >> 16
        reg = read_word_from_device(location, stream_reg(cb, 10), context=context) & 0xFFFF
        if local != reg:
            short.append(f"cb{cb} local={local} reg={reg}")
    if short:
        print(f"STUCK {location.to_user_str()} {' '.join(short)}", flush=True)
        state["stuck"].append(location)
    elif len(state["healthy"]) < 2:
        state["healthy"].append(location)


def snapshot(location):
    device = location._device
    block = location.noc_block
    out = {}
    store = block.get_register_store()
    gprs = device.get_tensix_registers_description().general_purpose_registers
    for t in range(3):
        for name, reg in gprs[t].items():
            try:
                out[f"gpr.t{t}.{name}"] = store.read_register(reg)
            except Exception as e:
                out[f"gpr.t{t}.{name}"] = f"ERR {e!r}"[:60]
    bus = block.debug_bus
    for name in sorted(bus.signal_names):
        if bus.is_combined_signal(name):
            continue
        try:
            out[f"dbus.{name}"] = bus.read_signal(name)
        except Exception as e:
            out[f"dbus.{name}"] = f"ERR {e!r}"[:60]
    return out


def run(args, context: Context):
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    get_run_checks(args, context).run_per_core_check(
        lambda location, risc_name: classify(location, risc_name, dispatcher_data, elfs_cache, context),
        block_filter=["tensix"],
        core_filter=["trisc2"],
    )
    snaps = {}
    for loc in state["stuck"] + state["healthy"]:
        tag = ("stuck " if loc in state["stuck"] else "healthy ") + loc.to_user_str()
        snaps[tag] = snapshot(loc)
        for k, v in snaps[tag].items():
            print(f"{tag} {k} = {v}", flush=True)
    healthy = [s for t, s in snaps.items() if t.startswith("healthy")]
    for tag, s in snaps.items():
        if not tag.startswith("stuck"):
            continue
        for k, v in s.items():
            others = [h.get(k) for h in healthy]
            if healthy and all(o != v for o in others):
                print(f"DIFF {tag} {k} = {v}  healthy = {others}", flush=True)


if __name__ == "__main__":
    run_script()
