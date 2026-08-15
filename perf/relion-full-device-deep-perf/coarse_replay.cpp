// coarse_replay -- replay RELION's own diff2_coarse on a dumped live call, at a chosen
// eulers_per_block, and time it.
//
// Why this instrument and not a refinement: one it13-17 refinement is ~15 min, and the question
// ("what does eulers_per_block cost") needs five points plus controls. The kernel is a pure
// function of the dumped arguments -- no MPI, no I/O, no sampling -- so replaying it answers the
// same question in seconds. The number it produces is a SINGLE-THREAD rate; the refinement runs 24
// threads sharing one L3, so the absolute ns/pair here is a floor, not the refinement's rate. Every
// arm is graded as a RATIO against E=1 measured on this same harness, and the winner is then
// confirmed in a full refinement. That is the split `tt-bio-isolated-op-timing-oversync-inflates-cost`
// asks for: screen isolated, price batched.
//
// Numerics: the output is written to a file so E arms can be compared with np.array_equal.

#include <limits>       // helper.h uses std::numeric_limits and gets it transitively in-tree

#include "src/acc/cpu/device_stubs.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/acc_projector.h"
#include "src/acc/cpu/cpu_kernels/helper.h"
#include "src/acc/cpu/cpu_kernels/diff2.h"
#include "src/acc/cpu/cpu_settings.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

struct Call {
	long g[13];
	std::vector<float> mdl, eul, tx, ty, img_r, img_i, w;
	int    mdlX()  const { return (int)g[0]; }
	int    mdlY()  const { return (int)g[1]; }
	int    mdlZ()  const { return (int)g[2]; }
	int    initY() const { return (int)g[3]; }
	int    initZ() const { return (int)g[4]; }
	int    maxR()  const { return (int)g[5]; }
	float  padf()  const { return (float)g[7]; }
	int    imgX()  const { return (int)g[8]; }
	int    imgY()  const { return (int)g[9]; }
	long   on()    const { return g[10]; }
	long   tn()    const { return g[11]; }
	long   is()    const { return g[12]; }
};

static void rd(FILE *fh, std::vector<float> &v, size_t n)
{
	v.resize(n);
	if (fread(v.data(), sizeof(float), n, fh) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

static Call load(const char *path)
{
	Call c;
	FILE *fh = fopen(path, "rb");
	if (!fh) { perror(path); exit(2); }
	if (fread(c.g, sizeof(long), 13, fh) != 13) { fprintf(stderr, "short header\n"); exit(2); }
	rd(fh, c.mdl,   (size_t)c.mdlX() * c.mdlY() * c.mdlZ() * 2);
	rd(fh, c.eul,   (size_t)c.on() * 9);
	rd(fh, c.tx,    (size_t)c.tn());
	rd(fh, c.ty,    (size_t)c.tn());
	rd(fh, c.img_r, (size_t)c.is());
	rd(fh, c.img_i, (size_t)c.is());
	rd(fh, c.w,     (size_t)c.is());
	fclose(fh);
	return c;
}

// One arm: RELION's own decomposition at eulers_per_block = E, i.e. exactly what
// runDiff2KernelCoarse would call if `rest` were taken modulo E instead of modulo blocks3D.
template<int E>
static void run_arm(Call &c, AccProjectorKernel &proj, std::vector<float> &out,
                    std::vector<float> &trans_z)
{
	const long rest = c.on() % E;
	const long even = c.on() - rest;
	if (even)
		CpuKernels::diff2_coarse<true, false, D2C_BLOCK_SIZE_REF3D, E, 4>(
			(unsigned long)(even / E), c.eul.data(),
			c.tx.data(), c.ty.data(), trans_z.data(),
			c.img_r.data(), c.img_i.data(), proj, c.w.data(),
			out.data(), (unsigned long)c.tn(), (unsigned long)c.is());
	if (rest)
		CpuKernels::diff2_coarse<true, false, D2C_BLOCK_SIZE_REF3D, 1, 4>(
			(unsigned long)rest, &c.eul[9 * even],
			c.tx.data(), c.ty.data(), trans_z.data(),
			c.img_r.data(), c.img_i.data(), proj, c.w.data(),
			&out[c.tn() * even], (unsigned long)c.tn(), (unsigned long)c.is());
}

typedef void (*armfn)(Call &, AccProjectorKernel &, std::vector<float> &, std::vector<float> &);

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: coarse_replay <call.bin> <E> [reps] [out.bin] [threads]\n");
		return 1;
	}
	Call c = load(argv[1]);
	const int E    = atoi(argv[2]);
	const int reps = argc > 3 ? atoi(argv[3]) : 3;

	AccProjectorKernel proj(
		c.mdlX(), c.mdlY(), c.mdlZ(),
		c.imgX(), c.imgY(), 1,
		c.initY(), c.initZ(),
		c.padf(), c.maxR(),
		(std::complex<XFLOAT> *)c.mdl.data());

	std::vector<float> trans_z((size_t)c.tn(), 0.0f);
	std::vector<float> out((size_t)c.on() * c.tn());
	const int nthr = argc > 5 ? atoi(argv[5]) : 1;

	armfn fn = nullptr;
	switch (E) {
		case 1:  fn = run_arm<1>;  break;
		case 2:  fn = run_arm<2>;  break;
		case 4:  fn = run_arm<4>;  break;
		case 8:  fn = run_arm<8>;  break;
		case 16: fn = run_arm<16>; break;
		case 32: fn = run_arm<32>; break;
		default: fprintf(stderr, "E must be 1,2,4,8,16,32\n"); return 1;
	}

	// Threaded mode reproduces the refinement's topology inside one rank: `nthr` threads each doing
	// a whole diff2_coarse call on its own particle, all sharing ONE 31.7 MB model volume against a
	// 16 MiB L3. That shared random gather is the term the single-thread rate cannot see, and it is
	// why the isolated rate below is a floor rather than the refinement's rate.
	std::vector<Call>               tc(nthr);
	std::vector<std::vector<float>> tout(nthr), tz(nthr);
	for (int i = 0; i < nthr; i++) {
		tc[i] = c;
		tc[i].mdl.clear();
		tc[i].mdl.shrink_to_fit();       // one model, shared via `proj`
		tout[i].assign((size_t)c.on() * c.tn(), 0.0f);
		tz[i].assign((size_t)c.tn(), 0.0f);
	}

	// One warm rep outside the timed loop: the model is 31.7 MB against a 16 MiB L3, so the first
	// pass pays cold-page faults the refinement never pays (it has been streaming the same buffer
	// for thousands of calls).
	#pragma omp parallel for num_threads(nthr) schedule(static)
	for (int i = 0; i < nthr; i++)
		fn(tc[i], proj, tout[i], tz[i]);

	double best = 1e30, sum = 0;
	for (int r = 0; r < reps; r++) {
		for (int i = 0; i < nthr; i++) std::fill(tout[i].begin(), tout[i].end(), 0.0f);
		auto t0 = std::chrono::steady_clock::now();
		#pragma omp parallel for num_threads(nthr) schedule(static)
		for (int i = 0; i < nthr; i++)
			fn(tc[i], proj, tout[i], tz[i]);
		auto t1 = std::chrono::steady_clock::now();
		double s = std::chrono::duration<double>(t1 - t0).count();
		if (s < best) best = s;
		sum += s;
	}
	out = tout[0];

	// nthr calls ran concurrently, so the per-pair rate divides by nthr as well.
	const double pairs = (double)c.on() * (double)c.is() * (double)nthr;
	printf("E=%-3d thr=%-3d reps=%d  best=%.4f s  mean=%.4f s  ns/pair=%.2f  on=%ld tn=%ld is=%ld\n",
	       E, nthr, reps, best, sum / reps, best * 1e9 / pairs, c.on(), c.tn(), c.is());

	if (argc > 4) {
		FILE *fh = fopen(argv[4], "wb");
		fwrite(out.data(), sizeof(float), out.size(), fh);
		fclose(fh);
	}
	return 0;
}
