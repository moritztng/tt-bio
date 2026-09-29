# Why the logs here end in `.logtxt`

`.gitignore:70` ignores `*.log` across the whole repo, so the allocator's own refusal -- the
only evidence behind the 512-token Wormhole ceiling in `docs/bindcraft2.md` -- was committed as
an empty hole: `round_events.json` landed and `round.log` did not. The finding would have been
a number in a document with nothing under it, on a leased box that gets reimaged.

Each `*.log` is therefore committed beside itself as `*.logtxt`, byte for byte. Nothing is
edited or summarised: a refusal is quotable only if it is the machine's words.
