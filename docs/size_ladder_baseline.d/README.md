Size-ladder baseline fragments.

Each file here is a whole `size_ladder_baseline.json` in the same format, holding one card
type's (or one record pass's) rows. The arm reads the monolith one directory up and then
overlays every fragment, last writer per (card, model) wins. Record into a fragment with:

    python3 scripts/release_gate.py --model size-ladder --size-ladder-record \
        --size-ladder-models protenix-v2 \
        --size-ladder-baseline docs/size_ladder_baseline.d/<card>.json

This exists so concurrent record passes on different cards and different models do not all
edit one shared JSON.
