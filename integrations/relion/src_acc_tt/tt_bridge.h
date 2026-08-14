#ifndef ACC_TT_BRIDGE_H_
#define ACC_TT_BRIDGE_H_

// Tenstorrent backend bridge. The ONLY file in RELION that knows about Python.
//
// The interface is deliberately plain C++ over plain arrays so that replacing the embedded-CPython
// implementation with a C++ libttnn one later needs no change anywhere else in RELION. Every entry
// point returns bool: false means "declined, use your own kernel", so an unsupported shape or a
// broken device degrades to the existing CPU path instead of failing the run.

namespace TTBridge
{
	// Coarse squared-difference kernel, 3D reference and 2D data.
	// mdlComplex is the padded Fourier reference, interleaved (re,im) per voxel.
	// diff2s is [orientation_num][translation_num], inner stride translation_num, ACCUMULATED onto.
	bool diff2Coarse(
			const float *mdlComplex,
			int mdlX, int mdlY, int mdlZ, int mdlInitY, int mdlInitZ,
			int maxR, int maxR2_padded, float padding_factor,
			int imgX, int imgY,
			const float *eulers,
			const float *trans_x, const float *trans_y,
			const float *Fimg_real, const float *Fimg_imag, const float *corr_img,
			float *diff2s,
			long orientation_num, long translation_num, long image_size);

	// Fine squared-difference kernel, 3D reference and 2D data. Same computation as the coarse
	// pass over a different, data-dependent orientation set; RELION asks for significant_num
	// entries of the orientation x translation matrix, named by rot_idx/trans_idx, and adds
	// sum_init to each. diff2s is [significant_num] and is ACCUMULATED onto.
	//
	// job_idx/job_num are RELION's CUDA block geometry and are deliberately not taken: the jobs
	// tile [0, significant_num) in order, so rot_idx/trans_idx already name every entry exactly
	// once (makeJobsForDiff2Fine, src/acc/acc_helper_functions_impl.h).
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
			long image_size, long job_num_count);

	// TT_RELION_CHECK=1 only. diff2Fine then computes into a private buffer and returns false, so
	// RELION runs its own kernel; this is called once RELION's kernel has written diff2s and
	// grades one against the other. Returns immediately when the mode is off.
	void diff2FineCheck(const float *diff2s, long significant_num);

	// Diagnostics: how many calls were handled on device and how many declined.
	void report();
}

#endif /* ACC_TT_BRIDGE_H_ */
