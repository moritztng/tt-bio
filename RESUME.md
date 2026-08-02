# tt-hw-planner-optimize-stress — resume state (2026-08-02 ~17:40 UTC)

Deliverable doc: /home/moritz/.coworker/state/tt-hw-planner-optimize-stress.md (on pc, keep updated).

## Done
- Branch f86e5baee9206e21ae158abb70bfe3b68ffcf084, source build OK (qb1:/home/ttuser/tt-metal-hwplanner),
  Tracy on, ttnn editable-installed over the wheel (uv pip install -e .), sfpi 7.67.0 at runtime/sfpi.
- Preconditions on card 1: PCC PASS (Top1 91.20/Top5 98.60 via direct demo node; shipped test_pcc.py
  deadlocks, issue #17). trace_replay 1cq WORKS: TRACE_PER_TOKEN_MS=24.1341 (probe w/o mode="auto";
  shipped test skips trace + false eager_terminal, issue #20).
- Issues filed: #17 #18 #19 #20 #21 (github.com/apande-TT/tt-metal/issues).
- Safety layers active on qb1 (do NOT remove while optimize runs): zz_tt_card_pin.pth +
  tt_shared_host_guard.py in hwplanner python_env site-packages; shim/tt-smi; drivers/stray_sweeper.sh
  (SWEEPER_ACTIVE), drivers/wt_sfpi_watcher.sh (WATCHER_ACTIVE).

## Running now
- drivers/optimize_run2.sh (detached): optimize llama3_1_8b_p150 --devices single --max-rounds 3,
  HF_MODEL set -> HF-referenced PCC gate generation. Log: logs/optimize_run2.log.
- Run 1 failed at step 5/10 (no gate; HF_MODEL unset). Supervisor restarted 3x on the deterministic
  error and attempted tt-smi -r 0 (blocked by shim) -> issue #21.

## Next (next relaunch)
1. Collect run 2 outcome: RUN_REPORT.md in the throwaway worktree (/tmp/tt_hw_planner_llama3_1_8b_p150_*),
   ledger at models/experimental/perf_automation/runs/<ts>/ledger.jsonl, PASS_TRACE markers in log.
2. Independent before/after: rerun fixed perf probe (TRACE_PER_TOKEN_MS) on HEAD vs the opt/* branch
   the engine commits wins to (git -C /home/ttuser/tt-metal-hwplanner branch -a, look for opt/).
   2-3 repetitions each, report median + spread; compare with the tool's claimed numbers.
3. If run 2 passes: raised-rounds run (--max-rounds 6) + a repeated run for determinism.
4. File the worktree runtime/sfpi isolation gap (watcher workaround) as an issue with evidence.
5. Update ~/tt-hw-planner-feedback.md + Dalar tracker (~/.hwplanner_xlsx_env/bin/python, BACKUP first),
   draft Slack reply, tg.sh checkpoint, finish state doc, run donecheck.
