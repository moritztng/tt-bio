# Pre-registered predictions, WH size sweep for K3 and K4
Written 2026-08-16T01:09Z, BEFORE the run. Op numbers from perf/whb2/out/divk_wh.json.

The 640 aa round (state doc 13.4) measured the isolated-per-op estimate overstating the fold
delta by 3.13x, not the ~2x the oversync lesson records. These predictions use BOTH: the raw
per-op product as the upper bound and that product / 3.13 as the point estimate.

Call count assumed 1120 per fold at 384, 448 and 576 (all below SEQ_LEN_MORE_CHUNKING = 608,
so the unchunked path, same as 384 and 512 in the census). 640 aa used the chunked path and
had 1680; if a size here reports a different count the prediction is void, not the result.

K4 at 384 aa   band k=64 2.8908 ms -> dividing k=192 1.8517 ms, saving 1.0391 ms/call
               upper 1.164 s, point 0.372 s, wall ~31.9 s  =>  1.2 % point, 3.6 % upper

K3 at 448 aa   stock 5.1197 ms -> fused at k=224 2.6410 ms, saving 2.4787 ms/call
               upper 2.776 s, point 0.887 s, wall ~40 s    =>  2.2 % point, 6.9 % upper

K3 at 576 aa   stock 16.0619 ms -> fused at k=192 5.4694 ms, saving 10.5925 ms/call
               upper 11.863 s, point 3.791 s, wall ~70 s   =>  5.4 % point, 17 % upper
               This is the largest predicted win on the document. 576 also had the largest
               op-level ratio, 2.9367x.

KILL GATE per size: if that size's two control arms spread wider than the A-vs-B delta,
report INCONCLUSIVE for that size and quote no ratio. Each size carries its own floor.
ACCURACY: neither lever is bit-exact, k_chunk sets the online-softmax reduction order.
pLDDT is the arm; digests are expected to differ between arms and be identical within one.
