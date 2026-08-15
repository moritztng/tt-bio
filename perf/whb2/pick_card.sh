# Pick a free card, sourced by every job on this box. Hard-coding one does not survive: four fleet
# workers now share UF-EV-A13-GWH02 and at least one takes its card outside the benchlock, so a card
# free at launch can be gone by the time a queued job reaches the front. UMD id = node + 24 over the
# range we are allowed to touch (audit: UMD 26-31 = /dev/tenstorrent/2..7; confirmed live twice this
# pass, node 2 opened as UMD 26 and node 3 was held by a worker running TT_VISIBLE_DEVICES=27).
# `sudo lsof /dev/tenstorrent/*` in the glob form is the only occupancy check on this box.
pick_card() {
  local n
  for n in 2 3 4 5 6 7; do
    if ! sudo lsof -t "/dev/tenstorrent/$n" >/dev/null 2>&1; then
      echo $((n + 24)); return 0
    fi
  done
  return 1
}
