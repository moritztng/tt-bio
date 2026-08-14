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

	// Diagnostics: how many calls were handled on device and how many declined.
	void report();
}

#endif /* ACC_TT_BRIDGE_H_ */
