# Installing the TT-Bio booth demo on a QuietBox 2

This turns a QuietBox 2 into the TT-Bio booth demo: it boots straight into a full-screen browser
where the four Blackhole chips fold proteins live, and repairs itself if a chip or the browser
fails. It installs the same kernel, driver, firmware check, software and model weights as the
QuietBox the demo was built and tested on, so it should behave the same way.

You do this once, a day or more before the event. It takes **under half an hour** plus a reboot,
most of it downloading. At the booth itself nothing needs installing.

## What you need

- The QuietBox 2, with Ubuntu as it came from the factory, and a user account that can use `sudo`.
- A screen with an HDMI input, cabled to the **HDMI port on the motherboard panel** (among the USB
  ports). The Tenstorrent cards have no display output.
- A USB keyboard.
- Network with internet access, **for the install only** (about 5 GB comes down: the code, two
  Python environments and the OpenFold3 weights). Once installed, the demo never uses the network.
- About 20 GB of free disk.

## Install

Log in on the box (at the screen, or over ssh) and run these two commands:

    git clone --depth 1 --branch booth-2026-10 https://github.com/moritztng/tt-bio ~/sc26
    ~/sc26/demo/sc26/install.py

The first takes under a minute. The second checks the box against the booth QuietBox and prints a
table, one line per part: kernel, driver, firmware, packages and so on. `ok` means it matches,
`fix` means the script will change it. Then it fixes what needs fixing; `sudo` asks for your
password once. Roughly:

| step | time |
|---|---|
| system packages, driver, kernel (only if different) | 0 to 10 min |
| two Python environments | a few minutes |
| OpenFold3 weights, 2.3 GB | a few minutes |
| recorded folds, settings, boot setup | under a minute |

On the QuietBox the demo was built on, a brand-new user account went from nothing to the end of
this list in 6 minutes. Each step prints how long it took.

If it stops part way (a dropped download, say), run `~/sc26/demo/sc26/install.py` again. It picks
up where it stopped and leaves alone whatever is already right.

When it finishes it says either **Reboot now to finish** or **Nothing to change**. Reboot with

    sudo reboot

and run `~/sc26/demo/sc26/install.py` once more after the reboot. It should end with
**Nothing to change: this box matches the booth QuietBox.** That is the sign the install is complete.

### If the script stops at the firmware

The script never updates the cards' firmware itself, because an interrupted firmware update can
leave a card unable to start. If the firmware differs from the booth box, it stops before changing
anything on the system, shows the versions, downloads and checks the right firmware file, and
prints the one command that installs it, with a warning. Run that command only when you can leave
the box alone and plugged in until it prints FLASH SUCCESS, then reboot and run the script again.
If the cards have **newer** firmware than the booth box, it says so; do not downgrade them, call
Moritz (below).

## First start

After the reboot the box logs in by itself and the demo appears. The first time takes longer,
because each chip compiles its programs once:

| after the reboot | on the screen |
|---|---|
| under 1 min | a still picture of a folded protein |
| about 1½ min | proteins folding, labelled **Recorded folds**; the chip rows say *warming up* |
| a few minutes, longer the very first time | **Live on four Blackhole chips**, each chip row folding |

From then on the box needs no keyboard and no login. `demo/sc26/BOOTH.md` is the one-page sheet
for the people at the booth (what the screen shows, what to do if it does not, whom to call).

## Check it is healthy

Press **Ctrl+Alt+F3** for a text login, log in, and run

    ~/sc26/demo/sc26/ops/sc26ctl health

It answers HEALTHY, REPAIRING ITSELF or NOT HEALTHY, one line per part, in plain words.
**Ctrl+Alt+F2** takes you back to the demo. Over ssh the same command works without leaving the
demo.

## The fallback video

If the live demo cannot be made to work at the booth, a 3½-minute recording of it plays on a loop
instead. The video is not public, so it does not come with the install: **ask Moritz for
`sc26-loop-4k-h264.mp4`** (294 MB) and put it on the box as

    ~/sc26-video/sc26-loop.mp4

Check the copy is complete:

    sha256sum ~/sc26-video/sc26-loop.mp4
    # 48bfc5ae82219bdd5addfcda37e64e98c4c1911df26c0cb2ae44a49408b72431

To switch to it: Ctrl+Alt+F3, log in, and

    ~/sc26/demo/sc26/ops/sc26ctl video

then Ctrl+Alt+F2 to see it. It plays full screen, without sound, until you go back to the live demo with

    ~/sc26/demo/sc26/ops/sc26ctl start

or reboot. Try this once after installing, so you know it works before you need it.

## If something goes wrong

| what happens | what to do |
|---|---|
| The install stops with an error | Run `~/sc26/demo/sc26/install.py` again. If it stops at the same place, send Moritz the last 30 lines it printed. |
| After the reboot the screen shows the normal Ubuntu login, not the demo | The install did not finish: log in, run `~/sc26/demo/sc26/install.py`, and reboot. |
| The demo runs but `sc26ctl health` says NOT HEALTHY for more than 15 minutes | Reboot once. If it is still not healthy, play the fallback video and call Moritz. |

To go back to a normal Ubuntu desktop after the event: `~/sc26/demo/sc26/ops/sc26ctl uninstall`, then reboot.

## Who to call

**Moritz Thüning**, mthuening@tenstorrent.com. Say what the screen or the script shows. If the box is
on the network, Moritz can usually fix things remotely.

More detail on how the demo runs and recovers: [ops/README.md](ops/README.md).
