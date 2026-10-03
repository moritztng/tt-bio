#!/bin/sh
# Keeps sway on exactly one place to draw. sway 1.9 aborts when a window maps while it has no
# output at all, which is what a box booted before its screen is plugged in (or switched on)
# has. So with no real screen sway gets a headless one, and the moment a real screen appears the
# headless one is unplugged and its workspace, browser included, moves onto the screen.
while :; do
  o=$(swaymsg -t get_outputs -r 2>/dev/null) || exit 0   # sway is gone; the session restarts us
  real=$(printf "%s" "$o" | grep -o "\"name\": *\"[^\"]*\"" | grep -vc HEADLESS)
  head=$(printf "%s" "$o" | grep -o "\"name\": *\"HEADLESS-[0-9]*\"" | grep -o "HEADLESS-[0-9]*")
  if [ "$real" -eq 0 ] && [ -z "$head" ]; then
    swaymsg -q create_output
  elif [ "$real" -gt 0 ] && [ -n "$head" ]; then
    for h in $head; do swaymsg -q output "$h" unplug; done
  fi
  sleep 2
done
