#ifdef _TT_ENABLED

#include "src/acc/tt/tt_bridge.h"

#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

// One interpreter per process. RELION runs one MPI rank per card, so one interpreter per card, and
// the GIL is not a throughput concern at --j 1. At --j > 1 the TBB worker threads serialise through
// PyGILState_Ensure, which is correct but slow; the batched path in a later pass removes the
// per-particle call entirely and with it the contention.

namespace
{
	std::mutex g_mutex;
	bool  g_tried    = false;   // init attempted
	bool  g_usable   = false;   // init succeeded
	PyObject *g_diff2 = nullptr;
	long  g_handled  = 0;
	long  g_declined = 0;

	void initLocked()
	{
		if (g_tried)
			return;
		g_tried = true;

		if (!Py_IsInitialized())
		{
			Py_InitializeEx(0);          // 0: do not install signal handlers, RELION owns them
			if (!Py_IsInitialized())
			{
				std::fprintf(stderr, "TTBridge: Py_InitializeEx failed, falling back to CPU\n");
				return;
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
			return;
		}
		g_diff2 = PyObject_GetAttrString(mod, "diff2_coarse");
		Py_DECREF(mod);
		if (g_diff2 == nullptr || !PyCallable_Check(g_diff2))
		{
			std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion.diff2_coarse missing\n");
			PyErr_Print();
			Py_XDECREF(g_diff2);
			g_diff2 = nullptr;
			PyGILState_Release(gil);
			return;
		}
		PyGILState_Release(gil);
		g_usable = true;
		std::fprintf(stderr, "TTBridge: tt_bio.cryoem.relion loaded\n");
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
	std::lock_guard<std::mutex> lock(g_mutex);
	initLocked();
	if (!g_usable)
	{
		g_declined++;
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
		g_usable = false;
	}
	else
	{
		handled = PyObject_IsTrue(res) == 1;
		Py_DECREF(res);
	}
	PyGILState_Release(gil);

	if (handled) g_handled++; else g_declined++;
	return handled;
}

void report()
{
	std::fprintf(stderr, "TTBridge: diff2Coarse handled=%ld declined=%ld\n", g_handled, g_declined);
}

}

#endif // _TT_ENABLED
