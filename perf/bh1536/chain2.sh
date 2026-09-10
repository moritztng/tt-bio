#!/bin/bash
# The second batch. Ordering is chain.sh's blocking flock on .card0.lock, so this no longer waits on
# another chain's process being gone -- that wait keyed on a cmdline, and the `bash -c ...
# setsid nohup ./chain2.sh &` launcher outlives its own `&` with the chain as its child, so the
# pattern kept matching and chain3.sh sat in its poll loop for 19 minutes without running one
# rung. Kept as a separate file only so a relaunch can see which batch is which in ps.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
exec ./perf/bh1536/chain.sh "$@"
