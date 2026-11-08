T1LW READ 11-08 03:3xZ (.108 chip 25, tree t8994fa4aa = main, --fast, 5 designs seed 0, rc 0 x11; logs
  .108:~/spd/spd-m-boltzgen/t1lw/wh-fast). md5 of the intermediate CIFs is 1ed90f7676 in EVERY arm, "all" included.
  AICLK on chip 25: 1000 MHz for every arm except b1 (129 of 343 samples at 810), so b2 is the reference.
  Trunk loop medians designs 2,3,5: b1 11.346, b2 11.342; trimul_t1b8 11.345, t1g8 11.339, t1_fgate 11.343, t1_part
  11.338, trimul_x8 11.344, dit_step_mm 11.335 (all dead: bytes and time equal); trimul_t1split 11.233 (-0.11, live but
  bit-identical, under the pre-registered -0.25 bar: KILLED); all 11.243 (= t1split). ms_step: b2 32.66, the trunk
  arms 31.5-32.3 (noise band), dit_shoist 41.09 and all 39.95 at clean 1000 MHz: dit_shoist costs WH +7-8 ms/step
  (KILLED; it only pays beside BH's trace). No keeper, so no stack A/B and no grade. Grid t1lw active: false.
  Bit-identical half of the flag space is now closed for BoltzGen on WH.

