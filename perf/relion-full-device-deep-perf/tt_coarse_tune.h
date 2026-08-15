// tt_coarse_tune.h -- runtime control and exact accounting for RELION's coarse diff2 dispatch.
//
// Two things, both behind -DTT_COARSE_TUNE so a build without the define is unchanged:
//
//  1. eulersPerBlock() reads TT_COARSE_E once, so one binary can run the whole
//     eulers-per-block sweep. Default 0 means "keep RELION's own behaviour exactly", which is what
//     the control arm runs.
//  2. An exact timer around the dispatch, accumulated atomically across threads and printed at
//     exit. The point is to separate the diff2_coarse KERNEL from the
//     getAllSquaredDifferencesCoarse REGION -- the region also builds the projector plan, generates
//     the eulers and does the weight bookkeeping, and no measurement in this program had ever
//     split them. Exact accounting rather than sampling, because the split is the whole question.

#ifndef TT_COARSE_TUNE_H_
#define TT_COARSE_TUNE_H_

#include <cstdio>
#include <cstdlib>
#include <omp.h>

namespace TTCoarseTune
{
	inline double &secs()  { static double d = 0;      return d; }
	inline double &pairs() { static double p = 0;      return p; }
	inline long   &calls() { static long   c = 0;      return c; }

	// 0 = leave RELION's dispatch alone (control). Otherwise the eulers-per-block to force.
	inline int eulersPerBlock()
	{
		static int e = -1;
		if (e < 0) {
			const char *s = getenv("TT_COARSE_E");
			e = s ? atoi(s) : 0;
			if (e != 0 && e != 1 && e != 2 && e != 4 && e != 8 && e != 16) {
				fprintf(stderr, "[tt_coarse_tune] TT_COARSE_E=%d not in {0,1,2,4,8,16}, using 0\n", e);
				e = 0;
			}
		}
		return e;
	}

	inline void report()
	{
		if (calls() == 0) return;
		fprintf(stderr, "[tt_coarse_tune] E=%d calls=%ld thread_s=%.3f pairs=%.4g ns/pair=%.2f\n",
		        eulersPerBlock(), calls(), secs(), pairs(),
		        pairs() > 0 ? secs() * 1e9 / pairs() : 0.0);
	}

	inline void account(double dt, double pr)
	{
		#pragma omp atomic
		secs() += dt;
		#pragma omp atomic
		pairs() += pr;
		#pragma omp atomic
		calls() += 1L;
		static bool once = (atexit(report), true);
		(void)once;
	}

	// One-shot on the first coarse call: does RELION's own modulus actually zero the blocked path
	// on this job? A static source read is not a measurement.
	inline void announce(unsigned long on, unsigned long blocks3D, int e)
	{
		static bool done = false;
		if (done) return;
		done = true;
		fprintf(stderr, "[tt_coarse_tune] first coarse call: orientation_num=%lu blocks3D=%lu "
		        "rest(%%blocks3D)=%lu even_orientation_num(RELION)=%lu  TT_COARSE_E=%d\n",
		        on, blocks3D, on % blocks3D, on - (on % blocks3D), e);
	}
}

#endif  // TT_COARSE_TUNE_H_
