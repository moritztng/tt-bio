# of3t-stepfloor — the RENORM prediction, registered before the paired rep ran

Written 2026-09-21T17:47Z, while arm A (`step_rekey_384.json`) is on rep 1 and has not yet
reached the OFF rep. Committed before the number exists, so it is a prediction and not a
reading dressed as one.

Two readings of `TT_BIO_SOFTMAX_BW_RENORM` exist at this point.

    scope                          ON          OFF        delta        arm
    trunk backward, ladder      357.32 s    356.00 s    +1.32 s     d164_probeoff_384 /
                                                        (+0.37 %)   d164_probeoff_renormoff_384
    full step, cold, process-A   585.062 s   685.579 s   -100.5 s    step_taped_384 /
                                             (-17.2 %)               step_renormoff_384

The second is impossible as a lever reading. The lever adds two ops inside a backward closure
and cannot make a step 100 s FASTER by being on. So the cross-process cold spread on this host
is larger than the effect, and the step-scope ON/OFF pair has to move inside one process.

PREDICTION for rep 2 of arm A, the OFF rep in a warm process that ran ON reps either side:
the step will come out FASTER with the lever off, by between 1 s and 10 s on a step of order
570 s, i.e. 0.2 % to 2 %. The basis is the trunk pair: 1.32 s of extra backward for the trunk's
share of the softmax-backward calls, and a step's tape is 9,888 nodes against the trunk's
2,473, so at most a few times the trunk's cost. Anything outside that band means either the
step's softmax-backward population is very different from the trunk's, or the in-process
spread is itself of that order, and both are findings.

The paired arms are `--renorm-per-rep 1,1,0,1` (arm A) and `0,0,1,0` (arm B).
