#!/usr/bin/env python3
"""tt-triage script: compute-side circular buffer counters on every Tensix core of a hung chip.

Per core and CB: TRISC0's local tiles_acked, TRISC2's local tiles_received and fifo_num_pages (the firmware global
cb_interface[] of each RISC), and the two stream-register counters the threads poll (Blackhole:
STREAM_REG_ADDR(cb, 10) = tiles_received, STREAM_REG_ADDR(cb, 8) = tiles_acked). A core whose pack thread waits for
space while its unpack thread waits for tiles shows here whether the packer's last tiles_received store landed.

usage (same environment and arguments as hangcap.sh's triage.py runs):
    PYTHONPATH=<triage dir>:... ~/triage_venv/bin/python dump_cb_counters.py -vv --inspector-rpc-port=... ...
"""
from ttexalens.context import Context
from ttexalens.memory_access import MemoryAccess
from ttexalens.tt_exalens_lib import read_word_from_device
from dispatcher_data import run as get_dispatcher_data
from elfs_cache import run as get_elfs_cache
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, run_script

script_config = ScriptConfig(depends=["run_checks", "dispatcher_data", "elfs_cache"])

NUM_CBS = 64


def stream_reg(stream, reg):
    return 0xFFB40000 + stream * 0x1000 + reg * 4


def field(entry, name):
    try:
        return int(getattr(entry, name).read_value())
    except Exception:
        try:
            return int(entry[name].read_value())
        except Exception:
            return None


def local_cbs(location, risc_name, dispatcher_data, elfs_cache):
    core = dispatcher_data.get_cached_core_data(location, risc_name)
    mem = MemoryAccess.create(location.noc_block.get_risc_debug(risc_name))
    cbi = elfs_cache[core.firmware_path].get_global("cb_interface", mem)
    return [cbi[cb] for cb in range(NUM_CBS)]


def dump(location, risc_name, dispatcher_data, elfs_cache, context):
    if risc_name != "trisc2":
        return
    try:
        unpack = local_cbs(location, "trisc0", dispatcher_data, elfs_cache)
        pack = local_cbs(location, "trisc2", dispatcher_data, elfs_cache)
    except Exception as e:
        log_check_location(location, False, f"cb_interface unreadable: {e}")
        return
    rows = []
    for cb in range(NUM_CBS):
        pages = field(pack[cb], "fifo_num_pages")
        if not pages:
            continue
        recv_reg = read_word_from_device(location, stream_reg(cb, 10), context=context) & 0xFFFF
        ack_reg = read_word_from_device(location, stream_reg(cb, 8), context=context) & 0xFFFF
        rows.append(f"cb{cb} pages={pages} pack.local_recv={field(pack[cb], 'tiles_received')} reg_recv={recv_reg} "
                    f"unpack.local_acked={field(unpack[cb], 'tiles_acked')} reg_acked={ack_reg}")
    log_check_location(location, True, "\n" + "\n".join(rows))


def run(args, context: Context):
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    get_run_checks(args, context).run_per_core_check(
        lambda location, risc_name: dump(location, risc_name, dispatcher_data, elfs_cache, context),
        block_filter=["tensix"],
        core_filter=["trisc2"],
    )


if __name__ == "__main__":
    run_script()
