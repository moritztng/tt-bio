#!/usr/bin/env bash
# Put a thin, lossy link between a browser on this box and the engine, without touching the
# engine's own loopback (the booth screen keeps its direct path).
#
#   thin_link.sh up <engine port> <rate> <delay> <loss>    e.g. up 8626 3mbit 40ms 1%
#   thin_link.sh shape <rate> <delay> <loss>                change the link while it is up
#   thin_link.sh bytes                                      bytes the link has carried, both ways
#   thin_link.sh down
#
# Then open http://10.77.0.2:<engine port>/app/ . A namespace `thinlink` is joined to the host by a
# veth pair; tc netem shapes the namespace's side, which carries every byte the engine sends to the
# browser, so the whole page (stream, folds, static files) crosses it. A relay on each side forwards
# 10.77.0.2:<port> -> 10.77.0.1:<port> -> 127.0.0.1:<port>. Each direction crosses the shaped side
# once, so the round trip is twice <delay>. Needs sudo; changes nothing outside the namespace and
# the two veth ends.
set -euo pipefail
NS=thinlink HOST=10.77.0.1 PEER=10.77.0.2
relay() {  # relay <listen host> <port> <target host> <port>: a plain TCP relay
  exec python3 -c '
import asyncio, sys
lh, lp, th, tp = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
async def pipe(r, w):
    try:
        while (b := await r.read(65536)):
            w.write(b); await w.drain()
    except (ConnectionError, OSError): pass
    finally: w.close()
async def conn(r, w):
    try: r2, w2 = await asyncio.open_connection(th, tp)
    except OSError: return w.close()
    await asyncio.gather(pipe(r, w2), pipe(r2, w))
async def main():
    async with await asyncio.start_server(conn, lh, lp): await asyncio.Event().wait()
asyncio.run(main())' "$@"
}
shape() { sudo ip netns exec $NS tc qdisc replace dev veth-n root netem rate "$1" delay "$2" loss "$3" limit 1000; }
case ${1:-} in
  up)
    port=$2
    sudo ip netns add $NS
    sudo ip link add veth-h type veth peer name veth-n
    sudo ip link set veth-n netns $NS
    sudo ip addr add $HOST/30 dev veth-h && sudo ip link set veth-h up
    sudo ip netns exec $NS ip addr add $PEER/30 dev veth-n
    sudo ip netns exec $NS ip link set veth-n up
    sudo ip netns exec $NS ip link set lo up
    shape "$3" "$4" "$5"
    (relay $HOST "$port" 127.0.0.1 "$port" >/dev/null 2>&1 &)
    (sudo ip netns exec $NS sudo -u "$(id -un)" bash "$0" _relay $PEER "$port" $HOST "$port" >/dev/null 2>&1 &)
    sleep 1; echo "http://$PEER:$port/app/  ($3, $4 each way, $5 loss)" ;;
  _relay) shift; relay "$@" ;;
  shape) shape "$2" "$3" "$4" ;;
  bytes) cat /sys/class/net/veth-h/statistics/rx_bytes /sys/class/net/veth-h/statistics/tx_bytes | paste -sd' ' ;;
  down)
    pkill -INT -f "asyncio.* $HOST [0-9]+ 127.0.0.1" || true
    sudo ip netns pids $NS 2>/dev/null | xargs -r sudo kill -INT || true
    sudo ip link del veth-h 2>/dev/null || true; sudo ip netns del $NS 2>/dev/null || true ;;
  *) sed -n 2,16p "$0"; exit 2 ;;
esac
