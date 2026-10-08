set -u
L=/home/moritz/.bci-seventeen-host/.bci/box_watch.log
exec >> "$L" 2>&1
K=$(cat /home/moritz/.config/vastai/vast_api_key)
I=54845453
echo "=== box_watch start $(date -u +%Y-%m-%dT%H:%M:%SZ) (poll 120s, relaunch clean on exit) ==="
# A stopped box loses the run: BindCraft 2 charges a claimed-but-unfinished trajectory, so we
# never resume a project folder. relaunch_host.sh moves the dead folder aside and starts clean.
while :; do
  s=$(curl -s -H "Authorization: Bearer $K" "https://console.vast.ai/api/v0/instances/" \
      | python3 -I -c "import sys,json;d=json.load(sys.stdin);i=[x for x in d.get('instances',d) if x.get('id')==$I];print(i[0].get('actual_status') if i else 'gone')")
  if [ "$s" != "running" ]; then
    echo "$(date -u +%H:%M:%SZ) state=$s -- relaunching clean"
    bash /home/moritz/.bci-seventeen-host/relaunch_host.sh
    echo "$(date -u +%H:%M:%SZ) relaunch returned"
  fi
  sleep 120
done
