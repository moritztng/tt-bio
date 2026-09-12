#!/usr/bin/env bash
# Arm the board's hardware watchdog so a kernel livelock resets the host instead of waiting for
# a human. Idempotent. Run as root. Verified on qb2 (AMD B850M-C, Ubuntu 24.04, 7.0.0-31-generic).
#
# The failure this is for: the host stops making progress without panicking (MINFRA-1586,
# PCB-4609 -- a long `mmput_async_fn` item on the bounded per-CPU `system_wq`). Nothing panics, so
# `panic_on_oops` and `hardlockup_panic` never fire and the box sits dead until someone walks over
# to it. A hardware watchdog needs the kernel to notice nothing: PID1 stops petting the timer and
# the board resets itself.
#
# Three things had to be got right, each of which silently produced a box with no watchdog while
# every config file read correct:
#
#  1. `sp5100_tco` is a platform driver. The `sp5100-tco` platform device it binds to is created by
#     `i2c-piix4` when that probes the AMD SMBus controller. Load `sp5100_tco` first and it binds
#     to nothing, exits 0, and never retries. Hence the softdep.
#  2. PID1 opens the watchdog device at manager startup, before the device exists on this board,
#     and only a `daemon-reexec` makes it look again -- `daemon-reload` does not. Hence the unit.
#  3. The unit must NOT be ordered `After=multi-user.target`. On qb2 that target is never reached,
#     because `plymouth-quit-wait.service` keeps running (measured: the target sat in
#     "start waiting" indefinitely), so anything ordered after it never runs at all. Being
#     `WantedBy=` the target is enough to be pulled in.
set -euo pipefail

TIMEOUT=${WATCHDOG_TIMEOUT:-60}        # systemd pets at TIMEOUT/2
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

cat > /etc/modprobe.d/sp5100-tco.conf <<'EOF'
softdep sp5100_tco pre: i2c_piix4
EOF
printf 'i2c_piix4\nsp5100_tco\n' > /etc/modules-load.d/sp5100-tco.conf

sed -i '/^i2c_piix4$/d;/^sp5100_tco$/d' /etc/initramfs-tools/modules
printf 'i2c_piix4\nsp5100_tco\n' >> /etc/initramfs-tools/modules

mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/99-qb2-watchdog.conf <<EOF
# A kernel livelock stops PID1 petting the watchdog and the board resets. ${TIMEOUT}s is well above
# any normal PID1 scheduling delay under a full-machine fold load, and well below a human.
[Manager]
RuntimeWatchdogSec=${TIMEOUT}
RebootWatchdogSec=2min
EOF

cat > /usr/local/sbin/arm-hardware-watchdog.sh <<'EOF'
#!/bin/bash
# Make sure PID1 is petting the board watchdog. Idempotent; safe to run at any time.
set -u
modprobe i2c_piix4 2>/dev/null || true
modprobe sp5100_tco 2>/dev/null || true
for _ in $(seq 1 30); do [ -e /dev/watchdog0 ] && break; sleep 1; done
if [ ! -e /dev/watchdog0 ]; then
  echo "<1>WATCHDOG-ARM FAILED: no /dev/watchdog0 after 30s; this host has no livelock recovery" > /dev/kmsg
  exit 1
fi
if ! systemctl show -p WatchdogDevice --value | grep -q .; then
  systemctl daemon-reexec       # the only thing that makes PID1 re-open the device
  sleep 3
fi
dev=$(systemctl show -p WatchdogDevice --value)
st=$(cat /sys/class/watchdog/watchdog0/state 2>/dev/null)
bs=$(cat /sys/class/watchdog/watchdog0/bootstatus 2>/dev/null)
# bootstatus 32 = WDIOF_CARDRESET: the previous boot ended because the watchdog reset the board.
# It is the only self-describing evidence that a death recovered on its own, and the witness on pc
# records this line over netconsole, so it survives the box.
echo "<1>WATCHDOG-ARM dev=${dev:-none} state=$st bootstatus=$bs$([ "$bs" = 32 ] && echo ' (PREVIOUS BOOT WAS A WATCHDOG RESET)')" > /dev/kmsg
[ -n "$dev" ] && [ "$st" = active ]
EOF
chmod +x /usr/local/sbin/arm-hardware-watchdog.sh

cat > /etc/systemd/system/arm-hardware-watchdog.service <<'EOF'
[Unit]
Description=Arm the board hardware watchdog against the kernel-livelock death class
After=systemd-modules-load.service
ConditionPathExists=/etc/systemd/system.conf.d/99-qb2-watchdog.conf

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/arm-hardware-watchdog.sh

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable arm-hardware-watchdog.service
update-initramfs -u
systemctl restart arm-hardware-watchdog.service

echo "--- verify ---"
systemctl show -p RuntimeWatchdogUSec -p WatchdogDevice
for f in identity state timeout timeleft bootstatus; do
  printf '%s=%s\n' "$f" "$(cat "/sys/class/watchdog/watchdog0/$f" 2>/dev/null)"
done
echo "holder: $(lsof /dev/watchdog0 2>/dev/null | awk 'NR==2{print $1" pid "$2}')"

# --- the boot splash, which is why the unit above is not ordered After=multi-user.target --------
#
# plymouth-quit.service runs `plymouth quit --retain-splash`, which keeps plymouthd alive. With no
# display manager to take over, plymouth-quit-wait.service never finishes, and two things break:
# multi-user.target is never reached, so any unit ordered after it never runs at all; and the
# console loglevel plymouth clamped to 1 is never restored, so a netconsole witness delivers
# nothing (ALERT is level 1 and the clamp needs level < 1, and every indicator still reads armed).
# Measured on qb2: `systemctl list-jobs` showed multi-user.target waiting on plymouth-quit-wait,
# `is-system-running` stuck at `starting`, and `kernel.printk` reading "1 4 1 7".
cat > /etc/systemd/system/plymouth-quit-headless.service <<'EOF'
[Unit]
Description=Dismiss the boot splash on a headless host
After=plymouth-quit.service systemd-user-sessions.service
ConditionPathExists=/bin/plymouth

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=-/bin/plymouth quit
ExecStartPost=/usr/bin/systemctl restart systemd-sysctl.service

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable plymouth-quit-headless.service
systemctl start plymouth-quit-headless.service
echo "system state: $(systemctl is-system-running)   printk: $(cat /proc/sys/kernel/printk)"
