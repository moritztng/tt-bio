#!/bin/sh
# systemd opens the watchdog device only when it starts, and the driver loads later in boot, so
# re-execute systemd once the device exists. Safe to run at any time.
modprobe i2c_piix4 2>/dev/null
modprobe sp5100_tco 2>/dev/null
for _ in $(seq 30); do [ -e /dev/watchdog0 ] && break; sleep 1; done
[ -e /dev/watchdog0 ] || { echo "<3>sc26: no /dev/watchdog0, the box cannot reboot itself after a hang" > /dev/kmsg; exit 1; }
[ -n "$(systemctl show -p WatchdogDevice --value)" ] || { systemctl daemon-reexec; sleep 3; }
[ "$(cat /sys/class/watchdog/watchdog0/state 2>/dev/null)" = active ]
