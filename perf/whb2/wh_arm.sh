# Shared arm runner for this task's Wormhole chains. Sourced by wh_k3_ab.sh and wh_split_v2.sh so
# the card-handling policy lives in one place.
#
# Two failure modes this box actually produces, both learned the hard way:
#
# 1. Another fleet worker takes the card between arms. A chain cannot pick one card at launch and
#    assume it still owns it; the device lease then refuses to open it (correctly, rather than
#    colliding at the fd level). So the card is picked per ARM and a lease loss is retried.
#
# 2. A card fails firmware init: "Timeout waiting for physical cores to finish ... failed to
#    initialize FW! Try resetting the board." Card 31 did this at 01:59:54Z on 2026-08-16 and took
#    an arm down with it. `pick_card` only checks lsof, so a wedged card looks free and would be
#    picked again and again. UF-EV-A13-GWH02 is production and shared with a customer, so resetting
#    it is not on the table: no fabric operations, ever. The card is quarantined for the rest of the
#    run instead and the arm retries elsewhere.
BAD_CARDS="${BAD_CARDS:-}"

pick_good_card() {
  local c
  for _ in 1 2 3 4 5 6; do
    c=$(pick_card) || return 1
    case " $BAD_CARDS " in *" $c "*) sleep 2; continue;; esac
    echo "$c"; return 0
  done
  return 1
}

# run_arm <label> <outdir> <retries> -- <xmodel_ab args...>
# Env passed through to the fold goes in ARM_ENV.
run_arm() {
  local label=$1 out=$2 retries=$3; shift 4
  local try C
  for try in $(seq 1 "$retries"); do
    C=$(pick_good_card) || { sleep 30; continue; }
    echo "=== $label try=$try card=$C $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
    env TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL \
        TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 ${ARM_ENV:-} \
      "$PY" perf/of3_4xpd/xmodel_ab.py --tree "$TREE" --label "$label" \
        --out "$out/$label.json" "$@" > "$out/$label.log" 2>&1
    if [ $? -eq 0 ]; then
      echo "EXIT $label = 0 (card $C)"
      grep -hE "median|cold " "$out/$label.log" | tail -2
      return 0
    fi
    if grep -q DeviceInUseError "$out/$label.log"; then
      sleep 15
    elif grep -qE "failed to initialize FW|waiting for physical cores to finish" "$out/$label.log"; then
      BAD_CARDS="$BAD_CARDS $C"
      echo "$label: card $C failed FW init, quarantined for this run (no reset: shared production box)"
    else
      echo "EXIT $label = FAILED, not a card problem"
      grep -iE "TT_THROW|TT_FATAL|Error:" "$out/$label.log" | head -3
      return 1
    fi
  done
  echo "EXIT $label = GAVE UP after $retries tries"
  return 1
}
