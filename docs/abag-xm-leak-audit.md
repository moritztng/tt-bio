# AbAg-XM leak audit (spec 1.4)

DBs: CDR-H3 from ANARCI over 26552 chains of 4610 ABAG-Rank-train SAbDab entries (opig TSV dead; pass-5 route); antigen leg vs all PDB protein chains (pdb_seqres + entries.idx). Flag rule: CDR-H3 >=90% AND antigen >=90% identity vs the SAME pre-cutoff entry.
Ranker leg: 0 of 164 targets appear in ABAG-Rank train complex_ids (asserted).

| leg | cutoff | median max-id | max max-id | n>=90% | n>=70% |
|---|---|---|---|---|---|
| CDR-H3 | 2021-09-30 | 0.0 | 100.0 | 60 | 63 |
| CDR-H3 | 2023-06-01 | 0.0 | 100.0 | 72 | 75 |
| antigen | 2021-09-30 | 96.5 | 100.0 | 123 | 146 |
| antigen | 2023-06-01 | 96.8 | 100.0 | 129 | 147 |
| antigen vs ABAG-Rank train | any (train) | 91.6 | 100.0 | 92 | 125 |

flagged (pre2021): 16 targets: 9l9o [6cbv,6usf,6ww2], 9mnb [6vvu], 9msc [6mej,6mek], 9n05 [7f4h], 9n09 [7f4h], 9n1p [7f4h], 9n2i [7f4h], 9pso [6xc4,6xc7,7d0c], 9tmp [6cbv,6usf,6ww2], 9u5p [6kyz], 9u5q [6kyz], 9u5r [6kyz], 9wb3 [6kyz], 9wb4 [6kyz], 9wpm [7f4h,7kh0,7rg9], 9yxd [1qfw]
flagged (pre2023): 34 targets: 21du [7t9i,7t9n,7tmw,7utz,7vuh,7vui,7vuj,7wu2,7wu3,7x2c,7x2d,7x2f,7xp4,7xp5,7xp6,7xw5,7xw6,8f76,8g2y,8hdo,8hdp,8hix,8hj0,8hj2,8hmp,8i2g,8irr], 9l9o [6cbv,6usf,6ww2,8hii,8hij,8hik], 9lxp [7srs,8as3,8go8,8gp3,8i0n,8i0q], 9lz0 [7srs,8as3,8go8,8gp3,8i0n,8i0q], 9lz1 [7srs,8as3,8go8,8gp3,8i0n,8i0q], 9m0j [8hdo,8hdp,8irr], 9m0x [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7xjh,7xji,8dcr,8dcs,8fu6,8gw8,8hdo,8hdp,8hmv,8irr], 9m0z [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wu2,7wu3,7xjh,7xji,8dcr,8dcs,8fu6,8hdo,8hdp,8hmv,8irr], 9m1p [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wu2,7wu3,7xjh,7xji,8dcr,8dcs,8fu6,8hdo,8hdp,8hmv,8irr], 9m2o [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7xjh,7xji,8dcr,8dcs,8fu6,8gw8,8hdo,8hdp,8hmv,8irr], 9m2s [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wu2,7wu3,7xjh,7xji,8dcr,8dcs,8fu6,8hdo,8hdp,8hmv,8irr], 9m3q [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wu2,7wu3,7xjh,7xji,8dcr,8dcs,8fu6,8hdo,8hdp,8hmv,8irr], 9m3s [7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wu2,7wu3,7xjh,7xji,8dcr,8dcs,8fu6,8hdo,8hdp,8hmv,8irr], 9m40 [8hdo,8hdp,8irr], 9mnb [6vvu], 9msc [6mej,6mek], 9mxu [8hdo,8hdp,8irr], 9mzf [8hdo,8hdp,8irr], 9n05 [7f4h,7wuj,7wxw,7wy8,7xke,7xv3,7ydp,8h4i,8ha0,8haf,8hao], 9n09 [7f4h,7vvj,7vvk,7vvl,7vvm,7vvn,7vvo,7wq4,7wuj,7wxu,7wxw,7wy5,7wy8,7xjh,7xji,7xke,7xv3,7ydm,7ydp,8gw8,8h4i,8h4k,8h4l,8ha0,8haf,8hao], 9n1p [7f4h,7wuj,7wxw,7wy8,7xke,7xv3,7ydp,8h4i,8ha0,8haf,8hao], 9n2i [7f4h,7wuj,7wxw,7wy8,7xke,7xv3,7ydp,8h4i,8ha0,8haf,8hao], 9n8i [7uvi], 9n8n [7unb,7uvi,7uxl], 9pso [7d0c,7u0q,7wcp], 9tmp [6cbv,6usf,6ww2,7tuy,8hii,8hij,8hik], 9u5p [6kyz], 9u5q [6kyz], 9u5r [6kyz], 9w14 [8iv5,8iv8], 9wb3 [6kyz], 9wb4 [6kyz], 9wpm [7f4h,7kh0,7rg9,7t9i,7tmw,7tyl,7tyn,7tyo,7tyx,7tyy,7tzf,7vqx,7vvm,7vvn,7vvo,7wbj,7wcn,7wu2,7wuj,7wuq,7wy5,7wy8,7x2c,7x2d,7x2f,7x8r,7x8s,7xjh,7xji,7xkd,7xke,7xkf,7xov,7xp4,7xp5,7xp6,7xtb,7xtc,7xv3,7xw5,7xw6,7xz5,7xz6,7y35,7y36,7y3g,7ydm,7ydp,7yp7,8dcr,8dcs,8e3z,8f0j,8f0k,8f2b,8f76,8flr,8fls,8flt,8flu,8fu6,8g2y,8gw8,8ha0,8haf,8hao,8hdo,8hdp,8hix,8hj0,8hmv,8irr], 9yxd [1qfw]
