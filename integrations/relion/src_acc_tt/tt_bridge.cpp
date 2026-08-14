#ifdef _TT_ENABLED

#include "src/acc/tt/tt_bridge.h"

#include <Python.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// One interpreter per process. RELION runs one MPI rank per card, so one interpreter per card.
//
// Only initialisation is mutexed. The call itself is not: PyGILState_Ensure already serialises the
// Python bytecode, and the tensor work underneath releases the GIL, so RELION's TBB worker threads
// overlap inside the heavy part. An earlier version held a std::mutex across the whole call, which
// serialised the threads a second time and independently of the GIL.

namespace
{
	std::mutex g_init_mutex;
	bool g_tried  = false;                  // init attempted, guarded by g_init_mutex
	std::atomic<bool> g_usable{false};      // read on every call, so atomic
	std::atomic<bool> g_fine_usable{false}; // separate: a fine-side bug must not silently switch
	                                        // the coarse arm back to CPU mid-refinement
	PyObject *g_diff2 = nullptr;            // set once under g_init_mutex, read-only after
	PyObject *g_diff2f = nullptr;
	std::atomic<long> g_handled{0};
	std::atomic<long> g_declined{0};
	std::atomic<long> g_fine_handled{0};
	std::atomic<long> g_fine_declined{0};

	bool g_check = false;                   // TT_RELION_CHECK, read once at init

	// TT_RELION_CHECK only. diff2Fine stashes its own answer here and declines; diff2FineCheck
	// grades it against RELION's once RELION's kernel has run. Thread-local because RELION calls
	// the kernel from every one of its --j worker threads.
	thread_local std::vector<float> t_ours;
	thread_local bool t_ours_valid = false;

	std::mutex g_res_mutex;
	double g_res_max = 0.0;                 // max |ours - relion| / |relion|
	long   g_res_n = 0;
	long   g_res_exact = 0;                 // entries that came out bit-identical anyway
	long   g_res_bucket[10] = {0};          // count by floor(-log10(relative residual))

	// Returns true when the Python entry points are ready to call.
	bool ensureReady()
	{
		if (g_usable.load(std::memory_order_acquire))
			return true;
		std::lock_guard<std::mutex> lock(g_init_mutex);
		if (g_tried)
			return g_usable.load(std::memory_order_acquire);
		g_tried = true;

		const char *chk = std::getenv("TT_RELION_CHECK");
		g_check = (chk != nullptr && chk[0] != '\0' && !(chk[0] == '0' && chk[1] == '\0'));

		if (!Py_IsInitialized())
		{
			Py_InitializeEx(0);          // 0: do not install signal handlers, RELION owns them
			if (!Py_IsInitialized())
			{
				std::fprintf(stderr, "TTBridge: Py_InitializeEx failed, falling back to CPU\n");
				return false;
			}
			PyEval_SaveThread();         // drop the GIL the main thread now holds
		}

		PyGILState_STATE gil = PyGILState_Ensure();
		PyObject *mod = PyImport_ImportModule("tt_bio.cryoem.relion");
		if (mod == nullptr)
		{
			std::fprintf(stderr, "TTBridge: cannot import tt_bio.cryoem.relion "
			                     "(is PYTHONPATH set to the tt-bio checkout?)\n");
			PyErr_Print();
			PyGILState_Release(gil);
			return false;
		}
		PyObject *fn = PyObject_GetAttrString(mod, "diff2_coarse");
		PyObject *fnf = PyObject_GetAttrString(mod, "diff2_fine");
		PyErr_Clear();                   // a tt_bio without the fine entry point is not an error
		Py_DECREF(mod);
		if (fn == nullptr || !PyCallable_Check(fn))
		{
			std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion.diff2_coarse missing\n");
			PyErr_Print();
			Py_XDECREF(fn);
			Py_XDECREF(fnf);
			PyGILState_Release(gil);
			return false;
		}
		PyGILState_Release(gil);
		g_diff2 = fn;
		g_usable.store(true, std::memory_order_release);
		if (fnf != nullptr && PyCallable_Check(fnf))
		{
			g_diff2f = fnf;
			g_fine_usable.store(true, std::memory_order_release);
		}
		else
		{
			Py_XDECREF(fnf);
			std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion.diff2_fine missing, "
			                     "the fine pass stays on RELION's own kernel\n");
		}
		std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion loaded (fine=%s check=%s)\n",
		             g_fine_usable.load() ? "yes" : "no", g_check ? "yes" : "no");
		return true;
	}
}

namespace TTBridge
{

bool diff2Coarse(
		const float *mdlComplex,
		int mdlX, int mdlY, int mdlZ, int mdlInitY, int mdlInitZ,
		int maxR, int maxR2_padded, float padding_factor,
		int imgX, int imgY,
		const float *eulers,
		const float *trans_x, const float *trans_y,
		const float *Fimg_real, const float *Fimg_imag, const float *corr_img,
		float *diff2s,
		long orientation_num, long translation_num, long image_size)
{
	if (!ensureReady())
	{
		g_declined.fetch_add(1);
		return false;
	}

	PyGILState_STATE gil = PyGILState_Ensure();

	const long nvox = (long)mdlX * (long)mdlY * (long)mdlZ;
	const Py_ssize_t f = (Py_ssize_t)sizeof(float);

	PyObject *a_mdl  = PyMemoryView_FromMemory((char*)mdlComplex, nvox * 2 * f, PyBUF_READ);
	PyObject *a_eul  = PyMemoryView_FromMemory((char*)eulers,     orientation_num * 9 * f, PyBUF_READ);
	PyObject *a_tx   = PyMemoryView_FromMemory((char*)trans_x,    translation_num * f, PyBUF_READ);
	PyObject *a_ty   = PyMemoryView_FromMemory((char*)trans_y,    translation_num * f, PyBUF_READ);
	PyObject *a_re   = PyMemoryView_FromMemory((char*)Fimg_real,  image_size * f, PyBUF_READ);
	PyObject *a_im   = PyMemoryView_FromMemory((char*)Fimg_imag,  image_size * f, PyBUF_READ);
	PyObject *a_corr = PyMemoryView_FromMemory((char*)corr_img,   image_size * f, PyBUF_READ);
	PyObject *a_out  = PyMemoryView_FromMemory((char*)diff2s,
	                                           orientation_num * translation_num * f, PyBUF_WRITE);

	PyObject *res = nullptr;
	if (a_mdl && a_eul && a_tx && a_ty && a_re && a_im && a_corr && a_out)
		res = PyObject_CallFunction(g_diff2, "OOOOOOOOiiiiiiifiilll",
				a_mdl, a_eul, a_tx, a_ty, a_re, a_im, a_corr, a_out,
				mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
				(double)padding_factor, imgX, imgY,
				orientation_num, translation_num, image_size);

	Py_XDECREF(a_mdl); Py_XDECREF(a_eul); Py_XDECREF(a_tx); Py_XDECREF(a_ty);
	Py_XDECREF(a_re);  Py_XDECREF(a_im);  Py_XDECREF(a_corr); Py_XDECREF(a_out);

	bool handled = false;
	if (res == nullptr)
	{
		// A Python-side failure is a bug, not a shape we decline. Print it once and stop trying, so
		// the run finishes on the CPU path instead of drowning the log.
		PyErr_Print();
		std::fprintf(stderr, "TTBridge: diff2_coarse raised, disabling the device path\n");
		g_usable.store(false, std::memory_order_release);
	}
	else
	{
		handled = PyObject_IsTrue(res) == 1;
		Py_DECREF(res);
	}
	PyGILState_Release(gil);

	if (handled) g_handled.fetch_add(1); else g_declined.fetch_add(1);
	return handled;
}

bool diff2Fine(
		const float *mdlComplex,
		int mdlX, int mdlY, int mdlZ, int mdlInitY, int mdlInitZ,
		int maxR, int maxR2_padded, float padding_factor,
		int imgX, int imgY,
		const float *eulers,
		const float *trans_x, const float *trans_y,
		const float *Fimg_real, const float *Fimg_imag, const float *corr_img,
		const unsigned long *rot_idx, const unsigned long *trans_idx,
		float *diff2s, float sum_init,
		long orientation_num, long translation_num, long significant_num,
		long image_size, long job_num_count)
{
	t_ours_valid = false;
	if (!ensureReady() || !g_fine_usable.load(std::memory_order_acquire) || significant_num <= 0)
	{
		g_fine_declined.fetch_add(1);
		return false;
	}

	// In check mode the answer goes into a private buffer and the call declines, so RELION's own
	// kernel still writes diff2s and diff2FineCheck has two independent answers to grade.
	float *dst = diff2s;
	if (g_check)
	{
		t_ours.assign((size_t)significant_num, 0.0f);
		dst = t_ours.data();
	}

	PyGILState_STATE gil = PyGILState_Ensure();

	const long nvox = (long)mdlX * (long)mdlY * (long)mdlZ;
	const Py_ssize_t f = (Py_ssize_t)sizeof(float);
	const Py_ssize_t u = (Py_ssize_t)sizeof(unsigned long);

	PyObject *a_mdl  = PyMemoryView_FromMemory((char*)mdlComplex, nvox * 2 * f, PyBUF_READ);
	PyObject *a_eul  = PyMemoryView_FromMemory((char*)eulers,     orientation_num * 9 * f, PyBUF_READ);
	PyObject *a_tx   = PyMemoryView_FromMemory((char*)trans_x,    translation_num * f, PyBUF_READ);
	PyObject *a_ty   = PyMemoryView_FromMemory((char*)trans_y,    translation_num * f, PyBUF_READ);
	PyObject *a_re   = PyMemoryView_FromMemory((char*)Fimg_real,  image_size * f, PyBUF_READ);
	PyObject *a_im   = PyMemoryView_FromMemory((char*)Fimg_imag,  image_size * f, PyBUF_READ);
	PyObject *a_corr = PyMemoryView_FromMemory((char*)corr_img,   image_size * f, PyBUF_READ);
	PyObject *a_ri   = PyMemoryView_FromMemory((char*)rot_idx,    significant_num * u, PyBUF_READ);
	PyObject *a_ti   = PyMemoryView_FromMemory((char*)trans_idx,  significant_num * u, PyBUF_READ);
	PyObject *a_out  = PyMemoryView_FromMemory((char*)dst,        significant_num * f, PyBUF_WRITE);

	PyObject *res = nullptr;
	if (a_mdl && a_eul && a_tx && a_ty && a_re && a_im && a_corr && a_ri && a_ti && a_out)
		res = PyObject_CallFunction(g_diff2f, "OOOOOOOOOOiiiiiiifiiflllll",
				a_mdl, a_eul, a_tx, a_ty, a_re, a_im, a_corr, a_ri, a_ti, a_out,
				mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
				(double)padding_factor, imgX, imgY, (double)sum_init,
				orientation_num, translation_num, significant_num, image_size, job_num_count);

	Py_XDECREF(a_mdl); Py_XDECREF(a_eul); Py_XDECREF(a_tx); Py_XDECREF(a_ty);
	Py_XDECREF(a_re);  Py_XDECREF(a_im);  Py_XDECREF(a_corr);
	Py_XDECREF(a_ri);  Py_XDECREF(a_ti);  Py_XDECREF(a_out);

	bool handled = false;
	if (res == nullptr)
	{
		PyErr_Print();
		std::fprintf(stderr, "TTBridge: diff2_fine raised, disabling the fine device path\n");
		g_fine_usable.store(false, std::memory_order_release);
	}
	else
	{
		handled = PyObject_IsTrue(res) == 1;
		Py_DECREF(res);
	}
	PyGILState_Release(gil);

	if (g_check)
	{
		t_ours_valid = handled;
		if (handled) g_fine_handled.fetch_add(1); else g_fine_declined.fetch_add(1);
		return false;                    // let RELION compute its own answer to grade against
	}

	if (handled) g_fine_handled.fetch_add(1); else g_fine_declined.fetch_add(1);
	return handled;
}

void diff2FineCheck(const float *diff2s, long significant_num)
{
	if (!g_check || !t_ours_valid || (long)t_ours.size() != significant_num)
		return;
	t_ours_valid = false;

	double mx = 0.0;
	long exact = 0;
	long bucket[10] = {0};
	for (long i = 0; i < significant_num; i++)
	{
		const double a = (double)t_ours[(size_t)i];
		const double b = (double)diff2s[i];
		if (t_ours[(size_t)i] == diff2s[i]) { exact++; continue; }
		const double den = std::fabs(b) > 0.0 ? std::fabs(b) : 1.0;
		const double r = std::fabs(a - b) / den;
		if (r > mx) mx = r;
		int k = (int)std::floor(-std::log10(r > 0.0 ? r : 1e-30));
		if (k < 0) k = 0;
		if (k > 9) k = 9;
		bucket[k]++;
	}

	std::lock_guard<std::mutex> lock(g_res_mutex);
	if (mx > g_res_max) g_res_max = mx;
	g_res_n += significant_num;
	g_res_exact += exact;
	for (int k = 0; k < 10; k++) g_res_bucket[k] += bucket[k];
}

void report()
{
	std::fprintf(stderr, "TTBridge: diff2Coarse handled=%ld declined=%ld\n",
	             g_handled.load(), g_declined.load());
	std::fprintf(stderr, "TTBridge: diff2Fine   handled=%ld declined=%ld\n",
	             g_fine_handled.load(), g_fine_declined.load());
	if (g_check && g_res_n > 0)
	{
		std::lock_guard<std::mutex> lock(g_res_mutex);
		std::fprintf(stderr, "TTBridge: fine residual n=%ld bit_identical=%ld max_rel=%.3e\n",
		             g_res_n, g_res_exact, g_res_max);
		std::fprintf(stderr, "TTBridge: fine residual by -log10(rel):");
		for (int k = 0; k < 10; k++) std::fprintf(stderr, " %d:%ld", k, g_res_bucket[k]);
		std::fprintf(stderr, "\n");
	}
}

}

#endif // _TT_ENABLED
