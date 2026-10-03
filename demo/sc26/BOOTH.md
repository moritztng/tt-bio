# TT-Bio at the booth

An AI predicts the 3D shape of a protein from its sequence, live, on the four Tenstorrent Blackhole
chips inside this QuietBox. Visitors type their name and watch it fold. Nobody needs to log in or
touch the box once it is on.

## Set up

1. **Place the box** with its vents clear. Do not open it.
2. **Screen:** HDMI cable from the box's **motherboard panel** (the HDMI port among the USB ports)
   to the screen. The Tenstorrent cards have no display output; plug nothing into their brackets.
   Switch the screen on and set its input to that HDMI. **Connect the screen before powering on.**
3. **Keyboard:** a USB keyboard in any USB port. This is what visitors type on. A touch screen
   also needs its USB cable.
4. **Network (optional):** booth Ethernet into the box's network port. The demo never uses it; it
   only lets us help remotely.
5. **Power:** plug in, press the power button once.

## What you should see

| after | on the screen |
|---|---|
| under 1 min | a still picture of a folded protein |
| ~1¼ min | points condensing into a protein, labelled **Recorded folds**; the four chip rows say *warming up* |
| ~1½ min | the label turns **Live on four Blackhole chips** and each chip row names what it is folding |

A chip row that says *recovering* or *resetting* for a few minutes is the box repairing that chip
while the others carry on; leave it. Visitors type letters and press Enter to fold; Esc goes back. **Tab** shows the four chips' clock,
power and temperature, Tab again returns. After a minute without input the screen goes back to
the loop by itself.

## If something goes wrong

| you see | do this |
|---|---|
| **No picture** | Check the cable sits in the motherboard's HDMI and the screen's input matches. If the screen was connected after the box was switched on and nothing appears within 10 seconds, restart it (below). |
| **A frozen or black screen** for more than 2 minutes | The box repairs itself within about a minute; if it has not after 2, restart it. |
| **Recorded folds** for more than 10 minutes, or every visitor gets *The chips are busy* | Leave it running: what it shows is real, recorded on this box, and labelled as such. Call. |

**Restart:** press the power button once, briefly. The box stops the demo cleanly (this can take
up to 2 minutes) and is back on it about 1½ minutes later. Do not hold the button: holding it
for 5 seconds cuts power, and then it stays off until someone presses it again.

**End of day:** leave the box running and switch only the screen off. If the hall cuts power
overnight, press the power button in the morning.

## Who to call

**Moritz Thüning**, phone ______________________, mthuening@tenstorrent.com.
Say what the screen shows and since when. With the Ethernet in, he can fix most things remotely.

<p class="foot">For whoever answers the call: <code>~/sc26/demo/sc26/ops/sc26ctl status | restart | logs</code>
as ttuser on qb2. A restart can take 90 s or more to stop the engine; never send it SIGKILL.
Details in <code>ops/README.md</code>; talking points in <code>TALKING.md</code>.</p>
