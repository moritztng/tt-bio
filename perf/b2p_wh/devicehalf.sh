#!/bin/bash
# b2p-wh device half, one sitting on .107 chip 30 (run from ~/b2p-wh/tt-bio).
R=${B2P_ROOT:-$HOME/b2p-wh/tt-bio}; PY=$HOME/bwx/venv/bin/python; OUT=$HOME/b2p-wh/out/dh2
T=tests/test_bindcraft2.py
TEN=""
for t in test_the_extra_msa_stack_runs_on_card_by_default test_the_extra_msa_stack_can_be_kept_in_jax \
  test_asking_for_the_extra_msa_swap_builds_one_on_the_evoformers_pool \
  test_the_campaign_path_can_ask_for_the_extra_msa_swap_too \
  test_the_gradient_leaves_softmax_and_layer_norm_on_the_device_by_default \
  test_the_exact_instrument_can_be_asked_for_off_explicitly \
  test_exact_false_arms_the_measured_round_and_puts_it_back \
  test_a_lever_env_var_still_takes_one_lever_out \
  test_the_campaign_path_carries_the_exact_argument_both_ways \
  test_the_control_arm_takes_one_gradient_step_from_tt_bio_alone; do TEN="$TEN $T::$t"; done
cmds=("cd $R && $PY -m pytest -q -p no:cacheprovider -rs $TEN")
for c in base gap gap-range ligands mse altloc mmcif twochain onechain peptide-short binder-long length-range no-hotspots; do
  cmds+=("cd $R && BCX_AF2=$HOME/bwx/af2_params $PY -u perf/bgx_inputs/round.py --case $c --rounds 2 --card 30 --out $OUT/rounds")
done
cmds+=("cd $R && $PY -m pytest -q -p no:cacheprovider -rs tests/test_bindcraft2.py tests/test_bcinputs.py")
cd $R && B2P_SCRIPT=sitting B2P_OUT_ROOT=$HOME/b2p-wh/out bash perf/b2p_wh/launch.sh dh2 30 --params $HOME/bwx/af2_params "${cmds[@]}"
