Size-ladder baseline fragments.

Each file here is one model's rows, keyed `<model>.json`, in the same
`cards.<card>.models.<model>` shape as `docs/size_ladder_baseline.json` one directory up.
`scripts/release_gate.py`'s size-ladder arm reads the monolith and overlays every fragment
here on top of it (`_size_ladder_read_baseline`); a model with no fragment is served from
the monolith exactly as before, and a model split across both (rows for one card in the
monolith, rows for another card in its fragment) reads as one merged entry.

Record into a fragment instead of the monolith with:

    python3 scripts/release_gate.py --model size-ladder --size-ladder-record \
        --size-ladder-fragment --size-ladder-models <model>

This exists so several models can be recorded in parallel on separate branches without all
editing one shared JSON — six workstreams landing rows in `size_ladder_baseline.json` at
once is a merge conflict (and a write race, if two ever run against the same checkout) by
construction; fragments never collide because each model owns exactly one file.
