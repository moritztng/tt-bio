# mol1-fast screen arms: ARM -> (--fast or not, TT_BIO_LEVERS). Source it, then `arm_env <arch> <arm>` sets FASTFLAG and LV.
# NORMAL_* is the shipped Mol-1 normal set on main 53688fdf1 (mode_levers + size_levers at >= 512 tokens on Blackhole),
# printed from the code, so an arm names its whole set (TT_BIO_LEVERS turns size_levers off).
NORMAL_wh=apb_ln,atom_k1,conf_bins,dit_chunk,dit_mm16,dit_qkv16,dit_qsplit,dit_sdpa,dit_sdpa32,dit_shoist_mm,dit_step_mm,msa_sample,opm_jd,pz_device,silu_f32,transition_ws,triatt_k1,triatt_ln,triatt_q10,triatt_tail,trimul_bres,trimul_ibw,trimul_t1_2g,trimul_t1_3g,trimul_t1_c256,trimul_t1_c320,trimul_t1_c384,trimul_t1_c448,trimul_t1_dram,trimul_t1_psig,trimul_tail,trimul_x8,trunk_hifi3,ws_chain,ws_l1acc,ws_psilu,ws_resid,ws_resid2
NORMAL_bh=adaln_mod,apb_ln,atom_k1,atom_l1,atom_mm16,conf_bins,dit_chunk,dit_mask_chain,dit_mm16,dit_qbox,dit_qkv16,dit_qkvf,dit_qsplit,dit_sdpa,dit_sdpa32,dit_shoist,edm_device,inproj_split,msa_sample,opm_jd,pz_device,silu_f32,transition_ws,triatt_k1,triatt_ln,triatt_tail,trimul_bres,trimul_ibw,trimul_t1_2g,trimul_t1_c256,trimul_t1_c320,trimul_t1_c384,trimul_t1_c448,trimul_t1_dram,trimul_t1_part,trimul_t1_psig,trimul_t1split,trimul_t2,trimul_t2cm,trimul_t2w,trimul_tail,trunk_hifi2,trunk_hifi3,ws_chain,ws_l1acc,ws_l1dest,ws_psilu,ws_resid,ws_resid2,transition_b8,ws_b8_plain,trimul_x8,trimul_t1w8,trimul_t1g8
# The bfp8 pair stack Protenix-v2 graded into its --fast sets, at Mol-1 widths (ungraded there).
B8_wh=transition_b8,ws_b8_plain,trimul_t1w8,trimul_t1g8,triatt_ln_b8,triatt_ln_b8g,trimul_gin_w8,opm_b8cfg,ws_ln,trimul_sig6,dit_mod16
B8_bh=triatt_ln_b8,triatt_ln_b8g,ws_ln,ws_lofi,trimul_sig6,dit_mod16
arm_env() {
  local n=NORMAL_$1 b=B8_$1
  case $2 in
    n)  FASTFLAG=; LV= ;;                        # shipped normal, no override
    f0) FASTFLAG=--fast; LV= ;;                  # shipped --fast today (generic set)
    b8) FASTFLAG=; LV=${!n},${!b} ;;             # normal mode + bfp8 pair stack
    fn) FASTFLAG=--fast; LV=${!n},fast ;;        # --fast over the normal set + shared fast set
    fb) FASTFLAG=--fast; LV=${!n},fast,${!b} ;;  # fn + bfp8 pair stack
    nf0) FASTFLAG=; LV=${!n},fast ;;            # fn without the old bfp8 _FAST_MODE: precision from the set alone
    nf) FASTFLAG=; LV=${!n},fast,${!b} ;;      # fb without _FAST_MODE
    *) return 1 ;;
  esac
}
