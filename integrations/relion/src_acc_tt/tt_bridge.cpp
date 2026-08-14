#ifdef _TT_ENABLED

#include "src/acc/tt/tt_bridge.h"

#include <Python.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

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
	PyObject *g_diff2 = nullptr;            // set once under g_init_mutex, read-only after
	std::atomic<long> g_handled{0};
	std::atomic<long> g_declined{0};

	// Returns true when the Python entry point is ready to call.
	bool ensureReady()
	{
		if (g_usable.load(std::memory_order_acquire))
			return true;
		std::lock_guard<std::mutex> lock(g_init_mutex);
		if (g_tried)
			return g_usable.load(std::memory_order_acquire);
		g_tried = true;

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
		Py_DECREF(mod);
		if (fn == nullptr || !PyCallable_Check(fn))
		{
			std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion.diff2_coarse missing\n");
			PyErr_Print();
			Py_XDECREF(fn);
			PyGILState_Release(gil);
			return false;
		}
		PyGILState_Release(gil);
		g_diff2 = fn;
		g_usable.store(true, std::memory_order_release);
		std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion loaded\n");
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

void report()
{
	std::fprintf(stderr, "TTBridge: diff2Coarse handled=%ld declined=%ld\n",
	             g_handled.load(), g_declined.load());
}

}

#endif // _TT_ENABLED
