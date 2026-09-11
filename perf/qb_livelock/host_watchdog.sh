#!/usr/bin/env bash
# Arm the board's hardware watchdog so a kernel livelock resets the host instead of waiting for
# a human.
#
# The failure this is for: the host stops making progress without panicking (MINFRA-1586,
# PCB-4609 -- a long `mmput_async_fn` item on the bounded per-CPU `system_wq`). Nothing panics, so
# `panic_on_oops` and `hardlockup_panic` never fire and the box sits dead. A hardware watchdog does
# not need the kernel to notice anything: PID1 stops petting the timer and the board resets itself.
#
# AMD boards expose the SP5100/SB800 TCO timer, which is not auto-probed -- hence modules-load.d.
# It also has to be in the initramfs, because PID1 opens the watchdog device before
# systemd-modules-load.service runs; without it systemd finds no device and silently stays off.
#
# Idempotent. Rebuilds the initramfs only when it changed something. Run as root.
set -euo pipefail

MODULE=${WATCHDOG_MODULE:-sp5100_tco}
TIMEOUT=${WATCHDOG_TIMEOUT:-60}          # systemd pets at TIMEOUT/2
changed=0

[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

modprobe "$MODULE"

printf '%s\n' "$MODULE" > "/etc/modules-load.d/${MODULE//_/-}.conf"

if ! grep -qx "$MODULE" /etc/initramfs-tools/modules; then
  printf '%s\n' "$MODULE" >> /etc/initramfs-tools/modules
  changed=1
fi

conf=/etc/systemd/system.conf.d/99-hardware-watchdog.conf
mkdir -p "$(dirname "$conf")"
new=$(cat <<CONF
# A kernel livelock stops PID1 petting the watchdog, and the board resets. $TIMEOUT s is well
# above any normal PID1 scheduling delay under a full-machine fold load and well below a human.
[Manager]
RuntimeWatchdogSec=$TIMEOUT
RebootWatchdogSec=2min
CONF
)
[ -f "$conf" ] && [ "$(cat "$conf")" = "$new" ] || { printf '%s\n' "$new" > "$conf"; changed=1; }

[ "$changed" = 1 ] && update-initramfs -u

# Arming PID1 live needs daemon-reexec, not daemon-reload: only a re-exec re-opens the device.
systemctl daemon-reexec

echo "--- verify ---"
lsinitramfs "/boot/initrd.img-$(uname -r)" | grep -c "$MODULE" | sed 's/^/module copies in initramfs: /'
systemctl show -p RuntimeWatchdogUSec -p WatchdogDevice
for f in identity state timeout timeleft nowayout bootstatus; do
  printf '%s=%s\n' "$f" "$(cat "/sys/class/watchdog/watchdog0/$f" 2>/dev/null)"
done
# bootstatus reads 32 (WDIOF_CARDRESET) on the boot that follows a watchdog-caused reset, which is
# the only self-describing evidence that a death actually self-recovered.
echo "holder: $(lsof /dev/watchdog0 2>/dev/null | awk 'NR==2{print $1" pid "$2}')"
