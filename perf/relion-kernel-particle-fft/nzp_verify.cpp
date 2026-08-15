// nzp_verify — prove that the fast direction loop about to be landed in
// HealpixSampling::selectOrientationsWithNonZeroPriorProbability is bit-identical to RELION's
// own, by linking RELION's UNPATCHED librelion_lib.a and comparing the two element by element.
//
// This is deliberately not the same program as nzp_screen. nzp_screen priced a rewrite that
// hoists q_j = M_j^T p out of the direction loop. That reordering is only bit-identical when
// M_j = L_j R_j^T is an exact signed permutation matrix (D2's are diagonal +-1, which is why it
// measured 0.000e+00). For a general point group the entries are irrational and the reorder can
// move the last ulp. What is landed therefore:
//
//   * gates the fast path on every M_j being an exact signed permutation, and
//   * accumulates the final dot product in RELION's own index order, from 0, so the arithmetic
//     is the identical sequence of IEEE operations rather than an algebraically equal one.
//
// Everything else falls through to RELION's original loop untouched.
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

// ============================================================ the cache that will become members
struct NzpCache
{
	std::vector<RFLOAT> dirx, diry, dirz;   // grid unit vectors, one per idir
	std::vector<int>    perm;               // 3 per operator: which component of d feeds output a
	std::vector<RFLOAT> sgn;                // 3 per operator: +1 or -1
	int  nsym = 0;
	bool usable = false;
};

// Build M_j = L_j R_j^T and require it to be an exact signed permutation matrix. Returns false
// the moment any operator is not, which switches the whole fast path off for this symmetry group.
static bool buildNzpCache(HealpixSampling &s, NzpCache &c)
{
	const size_t n = s.rot_angles.size();
	c.dirx.resize(n); c.diry.resize(n); c.dirz.resize(n);
	for (size_t i = 0; i < n; i++)
	{
		Matrix1D<RFLOAT> v;
		Euler_angles2direction(s.rot_angles[i], s.tilt_angles[i], v);
		c.dirx[i] = XX(v); c.diry[i] = YY(v); c.dirz[i] = ZZ(v);
	}

	c.nsym = (int)s.R_repository.size();
	c.perm.assign(3 * c.nsym, -1);
	c.sgn.assign(3 * c.nsym, 0.);
	for (int j = 0; j < c.nsym; j++)
	{
		bool colused[3] = {false, false, false};
		for (int a = 0; a < 3; a++)
		{
			int nz = 0;
			for (int b = 0; b < 3; b++)
			{
				RFLOAT acc = 0.;
				for (int k = 0; k < 3; k++)
					acc += MAT_ELEM(s.L_repository[j], a, k) * MAT_ELEM(s.R_repository[j], b, k);
				if (acc == 0.) continue;
				if (acc != 1. && acc != -1.) return false;   // not a signed permutation
				if (colused[b]) return false;                // two nonzeros in one column
				colused[b] = true;
				c.perm[3 * j + a] = b;
				c.sgn [3 * j + a] = acc;
				nz++;
			}
			if (nz != 1) return false;                       // row is empty or has two nonzeros
		}
	}
	c.usable = true;
	return true;
}

// ============================================================ the fast direction loop
// Preconditions enforced by the caller: is_3D, sigma_rot > 0, sigma_tilt > 0, !isRelax,
// !do_bimodal_search_psi, sigma_tilt_from_ninety <= 0, cache usable.
static void selectDirectionsFast(HealpixSampling &s, const NzpCache &c,
                                 RFLOAT prior_rot, RFLOAT prior_tilt,
                                 RFLOAT sigma_rot, RFLOAT sigma_tilt, RFLOAT sigma_cutoff,
                                 std::vector<int> &pointer_dir_nonzeroprior,
                                 std::vector<RFLOAT> &directions_prior)
{
	pointer_dir_nonzeroprior.clear();
	directions_prior.clear();

	Matrix1D<RFLOAT> prior_direction;
	Euler_angles2direction(prior_rot, prior_tilt, prior_direction);
	const RFLOAT px = XX(prior_direction), py = YY(prior_direction), pz = ZZ(prior_direction);
	const RFLOAT p[3] = {px, py, pz};

	// c_j[a] = p[a] * sign, so the fast dot is c[0]*d[perm0] + c[1]*d[perm1] + c[2]*d[perm2],
	// accumulated from 0 in index order exactly as dotProduct() does. Multiplying by +-1 is
	// exact, so folding the sign into p here cannot change a bit.
	std::vector<RFLOAT> cf(3 * c.nsym);
	for (int j = 0; j < c.nsym; j++)
		for (int a = 0; a < 3; a++)
			cf[3 * j + a] = p[a] * c.sgn[3 * j + a];

	const RFLOAT biggest_sigma = XMIPP_MAX(sigma_rot, sigma_tilt);
	const RFLOAT cut = sigma_cutoff * biggest_sigma;
	// Guard band: only directions whose dot clears cos(cut + 1e-6 deg) are given the real ACOSD,
	// and membership is then decided by RELION's own `diffang < cut` on that ACOSD. So the cosine
	// gate can never drop a boundary case, it only skips transcendentals that cannot matter.
	const double cos_gate = std::cos(((double)cut + 1e-6) * PI / 180.);

	RFLOAT sumprior = 0.;
	RFLOAT best_dot = -2.;
	long int best_idir = -999;
	const size_t n = c.dirx.size();

	for (size_t idir = 0; idir < n; idir++)
	{
		const RFLOAT d[3] = {c.dirx[idir], c.diry[idir], c.dirz[idir]};

		// identity operator: dotProduct(prior_direction, my_direction)
		RFLOAT acc = 0.;
		acc += p[0] * d[0];
		acc += p[1] * d[1];
		acc += p[2] * d[2];
		RFLOAT best = acc;

		for (int j = 0; j < c.nsym; j++)
		{
			const int   *pm = &c.perm[3 * j];
			const RFLOAT *cj = &cf[3 * j];
			RFLOAT a2 = 0.;
			a2 += cj[0] * d[pm[0]];
			a2 += cj[1] * d[pm[1]];
			a2 += cj[2] * d[pm[2]];
			if (a2 > best) best = a2;      // strict >, so first operator wins a tie, as RELION does
		}

		if (best > cos_gate)
		{
			RFLOAT diffang = ACOSD(best);
			if (diffang > 180.) diffang = ABS(diffang - 360.);   // unreachable, kept for fidelity
			if (diffang < cut)
			{
				pointer_dir_nonzeroprior.push_back((int)idir);
				RFLOAT prior = gaussian1D(diffang, biggest_sigma, 0.);
				sumprior += prior;
				directions_prior.push_back(prior);
			}
		}
		if (best > best_dot) { best_dot = best; best_idir = (long int)idir; }
	}

	for (size_t i = 0; i < directions_prior.size(); i++)
		directions_prior[i] /= sumprior;

	if (directions_prior.size() == 0)
	{
		pointer_dir_nonzeroprior.push_back((int)best_idir);
		directions_prior.push_back(1.);
	}
}

// ============================================================ driver
int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: nzp_verify <sampling.star> <healpix_order> <nreps>\n"); return 2; }
	const std::string fn = argv[1];
	const int order = atoi(argv[2]);
	const int nreps = atoi(argv[3]);

	HealpixSampling s;
	s.read(fn);
	s.healpix_order = order;
	s.initialise(3, false, false, false, false, false, 0., 0.);

	printf("sampling      : %s\n", fn.c_str());
	printf("healpix_order : %d\n", s.healpix_order);
	printf("sym           : %s   R_repository=%zu  isRelax=%d\n",
	       s.fn_sym.c_str(), s.R_repository.size(), (int)s.isRelax);
	printf("directions    : %zu\n", s.rot_angles.size());

	const RFLOAT step = s.getAngularSampling(false);
	const RFLOAT sigma = 2. * step;
	const RFLOAT sigma_cutoff = 3.;

	NzpCache c;
	double t0 = now_s();
	const bool ok = buildNzpCache(s, c);
	double t_ctx = now_s() - t0;
	printf("signed-perm gate : %s   (cache build %.3f ms, once per sampling change)\n",
	       ok ? "PASS -- fast path enabled" : "FAIL -- would fall back to RELION's loop", 1e3 * t_ctx);
	if (!ok) { printf("nothing to verify, fast path would not run\n"); return 0; }

	std::vector<RFLOAT> pr, pt;
	for (int i = 0; i < nreps; i++)
	{
		size_t k = (size_t)((double)i / nreps * s.rot_angles.size());
		pr.push_back(s.rot_angles[k] + 0.31);
		pt.push_back(s.tilt_angles[k] + 0.17);
	}

	std::vector<int> pdirA, pdirB, ppsi;
	std::vector<RFLOAT> dpriorA, dpriorB, psipriorA;

	// warm
	s.selectOrientationsWithNonZeroPriorProbability(pr[0], pt[0], 0., sigma, sigma, sigma,
	                                                pdirA, dpriorA, ppsi, psipriorA);
	selectDirectionsFast(s, c, pr[0], pt[0], sigma, sigma, sigma_cutoff, pdirB, dpriorB);

	size_t nmis_set = 0, nmis_val = 0, nkept = 0, ncmp = 0;
	double maxreldiff = 0.;
	for (int i = 0; i < nreps; i++)
	{
		s.selectOrientationsWithNonZeroPriorProbability(pr[i], pt[i], 0., sigma, sigma, sigma,
		                                                pdirA, dpriorA, ppsi, psipriorA);
		selectDirectionsFast(s, c, pr[i], pt[i], sigma, sigma, sigma_cutoff, pdirB, dpriorB);
		nkept += pdirA.size();
		bool setbad = (pdirA.size() != pdirB.size());
		if (!setbad)
			for (size_t k = 0; k < pdirA.size(); k++)
				if (pdirA[k] != pdirB[k]) { setbad = true; break; }
		if (setbad) { nmis_set++; continue; }
		for (size_t k = 0; k < dpriorA.size(); k++)
		{
			ncmp++;
			RFLOAT a = dpriorA[k], b = dpriorB[k];
			if (a != b)
			{
				nmis_val++;
				double rd = std::fabs((double)a - (double)b) / (std::fabs((double)a) + 1e-30);
				if (rd > maxreldiff) maxreldiff = rd;
			}
		}
	}
	printf("kept/call     : %.1f directions of %zu scanned\n", (double)nkept / nreps, s.rot_angles.size());
	printf("EXACTNESS     : index-set mismatches %zu/%d ; prior values compared %zu, mismatches %zu ; max rel diff %.3e\n",
	       nmis_set, nreps, ncmp, nmis_val, maxreldiff);

	t0 = now_s();
	for (int i = 0; i < nreps; i++)
		s.selectOrientationsWithNonZeroPriorProbability(pr[i], pt[i], 0., sigma, sigma, sigma,
		                                                pdirA, dpriorA, ppsi, psipriorA);
	double tA = now_s() - t0;

	t0 = now_s();
	for (int i = 0; i < nreps; i++)
		selectDirectionsFast(s, c, pr[i], pt[i], sigma, sigma, sigma_cutoff, pdirB, dpriorB);
	double tB = now_s() - t0;

	printf("RELION        : %10.3f ms/call   %8.1f ns/direction\n",
	       1e3 * tA / nreps, 1e9 * tA / nreps / s.rot_angles.size());
	printf("landed path   : %10.3f ms/call   %8.1f ns/direction\n",
	       1e3 * tB / nreps, 1e9 * tB / nreps / s.rot_angles.size());
	printf("SPEEDUP       : %.2fx  (direction loop only; the psi loop below it is untouched)\n", tA / tB);
	return 0;
}
