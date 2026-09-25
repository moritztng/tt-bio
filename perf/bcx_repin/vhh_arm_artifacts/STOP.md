# Stopped at a trajectory boundary, 2026-09-25T02:55:52Z

The stop conditional in state/ask-bcx-modelpool-decision.md: stop after trajectory 1 unless it reaches the final acceptance filter set. Trajectories 1, 2 and 3 all died at the SCREEN design stage on pLDDT against min_plddt_screen 0.60, candidates_scored 0, so the acceptance filters were never reached and the count cannot mean anything. Card 0 is owed to bcx-template.

`.campaign_state.json` now reads `trajectories: 10` so that
`CampaignProgress.claim_trajectory()` returns None and `campaign.py:152`'s loop breaks
at the next boundary. **3 trajectories were actually designed.**
The true ledger at the moment of the stop is `.campaign_state.pre_stop.json`, and the
record of what ran is `1_Trajectories/!_Trajectories.csv`, one row per trajectory.
No count in this campaign is reported from `.campaign_state.json`.
