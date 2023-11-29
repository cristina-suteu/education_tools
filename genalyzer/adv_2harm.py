import genalyzer
import genalyzer_advanced as gn
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle as MPRect

plot = True
navg = 2
nfft = 1024 * 256
fs = 1000
fsr = 2.0
ampl_dbfs = -1.0
freq = 1
phase = 0.0
td = 0.0
tj = 0.0
qres = 16  # quantizer resolution
qnoise_dbfs = -140.0  # quantizer noise
code_fmt = gn.CodeFormat.TWOS_COMPLEMENT
rfft_scale = gn.RfftScale.DBFS_SIN
window = gn.Window.NO_WINDOW

npts = navg * nfft
ampl = (fsr / 2) * 10 ** (ampl_dbfs / 20)
qnoise = 10 ** (qnoise_dbfs / 20)
ssb_fund = 4
ssb_rest = 3
if gn.Window.NO_WINDOW == window:
    freq = gn.coherent(nfft, fs, freq)
    ssb_fund = 0
    ssb_rest = 0

awf1 = gn.cos(npts, fs, ampl, freq, phase, td, tj)
awf2 = gn.cos(npts, fs, -2, 2, phase, td, tj)
awf = awf1 + awf2
pl.plot(awf[:10000])
pl.show()
qwf = gn.quantize(np.array(awf), fsr, qres, qnoise, code_fmt)
fft_cplx = gn.rfft(np.array(qwf), qres, navg, nfft, window, code_fmt, rfft_scale)

# Fourier analysis configuration
key = 'fa'
gn.mgr_remove(key)
gn.fa_create(key)
gn.fa_analysis_band(key, "fdata*0.0", "fdata*1.0")
gn.fa_fixed_tone(key, 'A', gn.FaCompTag.SIGNAL, freq, ssb_fund)
# gn.fa_max_tone(key, 'A', gn.FaCompTag.SIGNAL, ssb_fund)
gn.fa_hd(key, 3)
gn.fa_ssb(key, gn.FaSsb.DEFAULT, ssb_rest)
gn.fa_ssb(key, gn.FaSsb.DC, -1)
gn.fa_ssb(key, gn.FaSsb.SIGNAL, -1)
gn.fa_ssb(key, gn.FaSsb.WO, -1)
gn.fa_fsample(key, fs)
print(gn.fa_preview(key, False))

freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs)
fft_db = gn.db(fft_cplx)
fft_results = gn.fft_analysis(key, fft_cplx, nfft)
print("\nFourier Analysis Results:")
for k in ['fsnr', 'dc:mag_dbfs', 'A:freq', 'A:ffinal', 'A:mag_dbfs', 'A:phase',
          '2A:freq', '2A:ffinal', '2A:mag_dbfs', '2A:phase']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))

fftax = pl.subplot2grid((1, 1), (0, 0), rowspan=2, colspan=2)
pl.title("FFT")
pl.plot(freq_axis, fft_db)
pl.grid(True)
pl.xlim(freq_axis[0], 10)  # freq_axis[-1])
pl.ylim(-140.0, 20.0)
annots = gn.fa_annotations(fft_results)

for x, y, label in annots["labels"]:
    pl.annotate(label, xy=(x, y), ha='center', va='bottom')
for box in annots["tone_boxes"]:
    fftax.add_patch(MPRect((box[0], box[1]), box[2], box[3],
                           ec='pink', fc='pink', fill=True, hatch='x'))

pl.tight_layout()
pl.show()
