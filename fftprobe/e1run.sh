cd ~/.coworker/wt/ttnn-fft-kernel-spike
E1_BOX=128 E1_NPROJ=8000 E1_SNR=1.0 E1_CHUNK=200 python3 fftprobe/e1_fsc.py
E1_BOX=256 E1_NPROJ=4000 E1_SNR=1.0 E1_CHUNK=60  python3 fftprobe/e1_fsc.py
