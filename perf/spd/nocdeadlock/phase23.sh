# phase23.sh: VDD margin of board 4103 (chips 2,3, fw 19.15, process V/F curve, CG on) at 1350 MHz, same method as
# phase22 chip 1: pcG x1000 at forced VDD 800/790/780/770, stop at first HANG per chip. Decides whether board 4103
# also needs the 19.11 legacy curve. Index: "p23 margin chip=<c> vdd=<mV> verdict=", P23-DONE.
cd ~/spd/spd-nocdeadlock
for c in 2 3; do eval "exec $((6+c))>~/spd_qb2_chip$c.lock"; flock $((6+c)); done
echo "locks 2,3 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python
reset() { timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "reset 2,3 rc=$? $(date -u +%FT%TZ)"; sleep 15; }
reset
for c in 2 3; do
  for v in 800 790 780 770; do tag=p23_m${c}_$v
    echo "vdd $c $v: $(timeout -s TERM 60 $py vdd.py $c $v 2>&1 | tr "\n" " ")"
    ./repro.sh $c $tag fixm --iters 1000 --every 50
    d=$(grep -c REPRO-DONE rp_${tag}_c$c.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
    echo "p23 margin chip=$c vdd=$v verdict=$V last: $(grep "^it " rp_${tag}_c$c.log | tail -1) end $(date -u +%FT%TZ)"
    [ $V = HANG ] && break
  done
  timeout -s TERM 60 $py vdd.py $c 0 > /dev/null 2>&1; reset
done
echo "P23-DONE $(date -u +%FT%TZ)"
