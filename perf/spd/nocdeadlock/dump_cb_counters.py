#!/usr/bin/env python3
"""
Usage:
    dump_cb_counters

Description:
    Compute-side circular buffer counters on every Tensix core of a hung chip. Per core and CB: TRISC0's local
    tiles_acked, TRISC2's local tiles_received and fifo_num_pages (each RISC's firmware global cb_interface[]), and
    the stream-register counters the threads poll (Blackhole: STREAM_REG_ADDR(cb, 10) = tiles_received,
    STREAM_REG_ADDR(cb, 8) = tiles_acked). A core whose pack thread waits for space while its unpack thread waits for
    tiles shows here whether the packer's last tiles_received store landed. Run it standalone with hangcap.sh's
    environment and triage arguments.

Owner:
    spd-nocdeadlock2 (tt-bio)
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


_warned = set()


def field(entry, name):
    errs = []
    for get in (lambda: getattr(entry, name).read_value(), lambda: entry[name].read_value(), lambda: entry[name]):
        try:
            return int(get())
        except Exception as e:
            errs.append(repr(e))
    if name not in _warned:
        _warned.add(name)
        print(f"field {name} unreadable on {type(entry).__name__}: {errs}", flush=True)
    return None


def local_cbs(location, risc_name, dispatcher_data, elfs_cache):
    core = dispatcher_data.get_cached_core_data(location, risc_name)
    mem = MemoryAccess.create(location.noc_block.get_risc_debug(risc_name))
    cbi = elfs_cache[core.firmware_path].get_global("cb_interface", mem)
    return [cbi[cb].local_cb_interface for cb in range(NUM_CBS)]


def dump(location, risc_name, dispatcher_data, elfs_cache, context):
    if "dbg" not in _warned:
        _warned.add("dbg")
        print(f"first call: {location.to_user_str()} risc={risc_name!r}", flush=True)
    if str(risc_name) != "trisc2":
        return
    try:
        unpack = local_cbs(location, "trisc0", dispatcher_data, elfs_cache)
        pack = local_cbs(location, "trisc2", dispatcher_data, elfs_cache)
    except Exception as e:
        print(f"{location.to_user_str()} cb_interface unreadable: {e!r}", flush=True)
        return
    rows = []
    for cb in range(NUM_CBS):
        pages = field(pack[cb], "fifo_num_pages")
        if not pages:
            continue
        recv_reg = read_word_from_device(location, stream_reg(cb, 10), context=context) & 0xFFFF
        ack_reg = read_word_from_device(location, stream_reg(cb, 8), context=context) & 0xFFFF
        # tiles_acked / tiles_received are the low / high halves of one union word
        pack_word, unpack_word = field(pack[cb], "tiles_acked_received_init"), field(unpack[cb], "tiles_acked_received_init")
        rows.append(f"cb{cb} pages={pages} pack.local_recv={pack_word >> 16} reg_recv={recv_reg} "
                    f"unpack.local_acked={unpack_word & 0xFFFF} reg_acked={ack_reg}")
    for r in rows:
        print(f"{location.to_user_str()} {r}", flush=True)


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
