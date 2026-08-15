// nzp_screen — price RELION's own selectOrientationsWithNonZeroPriorProbability against a
// hoisted rewrite that produces the identical index list and priors.
//
// Why: the whole-refinement CTIC/CTOC profile says this call is 968.0 of the 1074.0 cumulative
// CPU-seconds inside getFourierTransformsAndCtfs (90.1%), i.e. ~42.7 s of the 922.19 s refinement,
// while the FFT that gave leg 3 its name is 34.6 CPU-s (~1.5 s). Before anyone builds anything,
// price the rewrite.
//
// The rewrite keeps the selection bit-identical:
//   sym_direction = L_j * (d^T R_j)^T = (L_j R_j^T) d, so M_j = L_j R_j^T is loop-invariant, and
//   dot(p, M_j d) = dot(M_j^T p, d), so q_j = M_j^T p is invariant over the direction loop too.
//   The grid unit vectors are invariant over particles. ACOSD is strictly decreasing, so argmax
//   dot == argmin diffang with identical tie order. The cutoff test still runs on the real ACOSD
//   for every candidate inside a guard band, so the kept set is the same set, not an approximation.
#include <src/healpix_sampling.h>
#include <src/euler.h>
#include <src/time.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static double now_s()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// ---------------------------------------------------------------- the rewrite
struct FastCtx
{
	std::vector<double> dx, dy, dz;      // grid unit vectors, one per idir
	std::vector<double> M;               // 9 doubles per symmetry operator, M_j = L_j R_j^T
	int nsym = 0;
};

static void buildCtx(HealpixSampling &s, FastCtx &c)
{
	const size_t n = s.rot_angles.size();
	c.dx.resize(n); c.dy.resize(n); c.dz.resize(n);
	for (size_t i = 0; i < n; i++)
	{
		Matrix1D<RFLOAT> v;
		Euler_angles2direction(s.rot_angles[i], s.tilt_angles[i], v);
		c.dx[i] = XX(v); c.dy[i] = YY(v); c.dz[i] = ZZ(v);
	}
	c.nsym = (int)s.R_repository.size();
	c.M.assign(9 * c.nsym, 0.);
	for (int j = 0; j < c.nsym; j++)
	{
		// M = L * R^T
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
			{
				double acc = 0.;
				for (int k = 0; k < 3; k++)
					acc += MAT_ELEM(s.L_repository[j], a, k) * MAT_ELEM(s.R_repository[j], b, k);
				c.M[9 * j + 3 * a + b] = acc;
			}
	}
}

static void selectFast(HealpixSampling &s, const FastCtx &c,
                       RFLOAT prior_rot, RFLOAT prior_tilt,
                       RFLOAT sigma_rot, RFLOAT sigma_tilt, RFLOAT sigma_cutoff,
                       std::vector<int> &pdir, std::vector<RFLOAT> &dprior,
                       long int &best_idir_out)
{
	pdir.clear(); dprior.clear();
	Matrix1D<RFLOAT> pv;
	Euler_angles2direction(prior_rot, prior_tilt, pv);
	const double px = XX(pv), py = YY(pv), pz = ZZ(pv);

	// q_0 = p (identity), q_j = M_j^T p
	std::vector<double> qx(1 + c.nsym), qy(1 + c.nsym), qz(1 + c.nsym);
	qx[0] = px; qy[0] = py; qz[0] = pz;
	for (int j = 0; j < c.nsym; j++)
	{
		const double *M = &c.M[9 * j];
		qx[1 + j] = M[0] * px + M[3] * py + M[6] * pz;
		qy[1 + j] = M[1] * px + M[4] * py + M[7] * pz;
		qz[1 + j] = M[2] * px + M[5] * py + M[8] * pz;
	}

	const double biggest_sigma = XMIPP_MAX(sigma_rot, sigma_tilt);
	const double cut_deg = sigma_cutoff * biggest_sigma;
	// guard band: only directions whose dot clears cos(cut + 1e-6 deg) get the real ACOSD test,
	// so the comparison that decides membership is byte-for-byte RELION's own.
	const double cos_gate = std::cos((cut_deg + 1e-6) * PI / 180.);

	double sumprior = 0., best_dot = -2.;
	long int best_idir = -999;
	const size_t n = c.dx.size();
	for (size_t idir = 0; idir < n; idir++)
	{
		const double ddx = c.dx[idir], ddy = c.dy[idir], ddz = c.dz[idir];
		double best = qx[0] * ddx + qy[0] * ddy + qz[0] * ddz;
		for (int j = 1; j <= c.nsym; j++)
		{
			const double v = qx[j] * ddx + qy[j] * ddy + qz[j] * ddz;
			if (v > best) best = v;
		}
		if (best > cos_gate)
		{
			const double diffang = ACOSD(best);
			if (diffang < cut_deg)
			{
				pdir.push_back((int)idir);
				const double prior = gaussian1D(diffang, biggest_sigma, 0.);
				dprior.push_back(prior);
				sumprior += prior;
			}
		}
		if (best > best_dot) { best_dot = best; best_idir = (long int)idir; }
	}
	best_idir_out = best_idir;
	if (sumprior > 0.)
		for (size_t i = 0; i < dprior.size(); i++) dprior[i] /= sumprior;
}

// ---------------------------------------------------------------- driver
int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: nzp_screen <sampling.star> <healpix_order> <nreps>\n"); return 2; }
	const std::string fn = argv[1];
	const int order = atoi(argv[2]);
	const int nreps = atoi(argv[3]);

	HealpixSampling s;
	s.read(fn);
	s.healpix_order = order;
	s.initialise(3, false, false, false, false, false, 0., 0.);
	printf("sampling      : %s\n", fn.c_str());
	printf("healpix_order : %d\n", s.healpix_order);
	printf("sym           : %s   R_repository=%zu  L_repository=%zu  isRelax=%d\n",
	       s.fn_sym.c_str(), s.R_repository.size(), s.L_repository.size(), (int)s.isRelax);
	printf("directions    : %zu     psi samples: %zu\n", s.rot_angles.size(), s.psi_angles.size());

	// RELION's own value at this order: sigma2_rot = (2 * rottilt_step)^2  (ml_optimiser.cpp:2324)
	const RFLOAT step = s.getAngularSampling(false);
	const RFLOAT sigma = 2. * step;
	const RFLOAT sigma_cutoff = 3.;
	printf("rottilt_step  : %g deg -> sigma %g deg, cutoff %g deg\n",
	       (double)step, (double)sigma, (double)(sigma_cutoff * sigma));

	FastCtx ctx;
	double t0 = now_s();
	buildCtx(s, ctx);
	double t_ctx = now_s() - t0;
	printf("ctx build     : %.3f ms   (once per sampling change, not per particle)\n", 1e3 * t_ctx);

	// realistic priors: walk the grid itself, jittered, so the survivor counts are representative
	std::vector<RFLOAT> pr, pt;
	for (int i = 0; i < nreps; i++)
	{
		size_t k = (size_t)((double)i / nreps * s.rot_angles.size());
		pr.push_back(s.rot_angles[k] + 0.31);
		pt.push_back(s.tilt_angles[k] + 0.17);
	}

	std::vector<int> pdirA, pdirB, ppsi;
	std::vector<RFLOAT> dpriorA, dpriorB, psipriorA, psipriorB;
	long int bidir;

	// warm both
	s.selectOrientationsWithNonZeroPriorProbability(pr[0], pt[0], 0., sigma, sigma, sigma,
	                                                pdirA, dpriorA, ppsi, psipriorA);
	selectFast(s, ctx, pr[0], pt[0], sigma, sigma, sigma_cutoff, pdirB, dpriorB, bidir);

	// ---- exactness over every rep
	size_t nmis_set = 0, nmis_val = 0, nkept = 0;
	double maxreldiff = 0.;
	for (int i = 0; i < nreps; i++)
	{
		s.selectOrientationsWithNonZeroPriorProbability(pr[i], pt[i], 0., sigma, sigma, sigma,
		                                                pdirA, dpriorA, ppsi, psipriorA);
		selectFast(s, ctx, pr[i], pt[i], sigma, sigma, sigma_cutoff, pdirB, dpriorB, bidir);
		nkept += pdirA.size();
		if (pdirA.size() != pdirB.size()) { nmis_set++; continue; }
		for (size_t k = 0; k < pdirA.size(); k++)
		{
			if (pdirA[k] != pdirB[k]) { nmis_set++; break; }
		}
		for (size_t k = 0; k < dpriorA.size() && k < dpriorB.size(); k++)
		{
			double a = dpriorA[k], b = dpriorB[k];
			if (a != b)
			{
				nmis_val++;
				double rd = std::fabs(a - b) / (std::fabs(a) + 1e-30);
				if (rd > maxreldiff) maxreldiff = rd;
			}
		}
	}
	printf("kept/call     : %.1f directions of %zu scanned\n", (double)nkept / nreps, s.rot_angles.size());
	printf("EXACTNESS     : index-set mismatches %zu/%d ; prior-value mismatches %zu ; max rel diff %.3e\n",
	       nmis_set, nreps, nmis_val, maxreldiff);

	// ---- timing, RELION's own
	t0 = now_s();
	for (int i = 0; i < nreps; i++)
		s.selectOrientationsWithNonZeroPriorProbability(pr[i], pt[i], 0., sigma, sigma, sigma,
		                                                pdirA, dpriorA, ppsi, psipriorA);
	double tA = now_s() - t0;

	t0 = now_s();
	for (int i = 0; i < nreps; i++)
		selectFast(s, ctx, pr[i], pt[i], sigma, sigma, sigma_cutoff, pdirB, dpriorB, bidir);
	double tB = now_s() - t0;

	printf("RELION        : %10.3f ms/call   %8.1f ns/direction\n",
	       1e3 * tA / nreps, 1e9 * tA / nreps / s.rot_angles.size());
	printf("rewrite       : %10.3f ms/call   %8.1f ns/direction\n",
	       1e3 * tB / nreps, 1e9 * tB / nreps / s.rot_angles.size());
	printf("SPEEDUP       : %.2fx  (direction loop only; the psi loop below it is untouched)\n", tA / tB);
	return 0;
}
