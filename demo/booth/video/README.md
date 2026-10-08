# The demo video

This branch, web-video-notime-2026-10, makes the Tenstorrent website's cut without any times: the title
reads "Biology Models on Tenstorrent", there are no QR codes and no "local" marks, and no fold time, stopwatch
or replay speed appears anywhere. The number under each protein names the model that folded it, and each chip
row ends on the chip's stage counter. The cut with times is web-video-2026-10; the booth's own screen is on
booth-2026-10.

A recorder for a looping video of the booth screen with all four chip lanes busy. The booth itself
never runs anything here: the kiosk serves `web/` and nothing else, and no file in `web/`, `engine/`
or `ops/` refers to this directory.

Chip 2 of the QuietBox is out of service, so the live demo folds on three chips. The video shows four.
Every fold in it is a real one: the booth engine ran it on chip 0, 1 or 3 of this box at 1350 MHz,
and `capture.py` recorded its stream messages as they arrived and its coordinates. Lanes 1, 2 and 4
show folds of chips 0, 1 and 3. Lane 3 shows folds those chips ran at other times. Which lane a fold
appears in is the only thing staged.

| step | what |
|------|------|
| `capture.py` | listens to the running engine like any page and keeps its messages and finished folds |
| `compose.py plan` | picks each lane's folds: at most three chips on long folds, no protein on two lanes at once, all eleven proteins every loop |
| `compose.py build` | writes the session: the folds' own messages at their own times, repeating exactly every period |
| `render.py` | serves `web/` unchanged plus `shim.js`, and records the page in Firefox on qb2's GPU, one finished frame at a time |
| `compose.py seam` | finds the loop's cut: two stage changes one period apart showing the same fold |

`shim.js` runs the page on a virtual clock that moves one video frame per step and waits until the
frame is complete (fold pulls, the mesh worker, CSS transitions) before it is taken, so there are no
dropped or half-drawn frames, and the same session renders the same frames every time.

    python3 capture.py --out ~/booth-video/tap --minutes 90
    python3 compose.py plan  --out ~/booth-video/s1 --period 212
    python3 compose.py build --out ~/booth-video/s1 --period 212
    python3 render.py --session ~/booth-video/s1 --size 1280x720 --end 57300   # timing run
    python3 compose.py seam  --out ~/booth-video/s1                            # gives the cut
    python3 compose.py build --out ~/booth-video/s1 --period <cut frames / 60>
    python3 render.py --session ~/booth-video/s1 --from <k> --to <k + frames> --end <k + frames> --out frames.mkv

The loop is about 211 s because the stage brings the same protein back every five to six turns, and a
page fetches a protein's newer fold at most every 120 s (`stream.js`). A shorter loop could not end
on the frame it starts with.
