import genalyzer_advanced as gn
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle as MPRect

# Configuration Params
navg = 2  # no. of fft averages
nfft = 1024 * 256  # FFT order
# get number of points
npts = navg * nfft
fs = 1000  # sampling frequency in Hz

phase = 0.0  # tone phase
td = 0.0
tj = 0.0
qres = 16  # quantizer resolution
qnoise_dbfs = -140.0  # quantizer noise
code_fmt = gn.CodeFormat.TWOS_COMPLEMENT  # ADC codes format
rfft_scale = gn.RfftScale.DBFS_SIN  # fft scale
window = gn.Window.BLACKMAN_HARRIS  # fft window

fsr = 2.0  # full-scale range
fund_freq = 1.0  # Hz

# harm_dbfs = [3.0]
# Uncomment this line to add more harmonics
harm_dbfs = [-3.0, -20.0, -40.0, -60.0]  # tone amplitude in dbFS


noise_freqs = [1.5, 2.5, 3.5, 4.5]
noise_dbfs = [-75, -80, -85, -90]

# Calculate amplitudes from dBfs
harm_ampl = []
for x in range(len(harm_dbfs)):
    harm_ampl.append((fsr / 2) * 10 ** (harm_dbfs[x] / 20))
noise_ampl = []
for x in range(len(noise_dbfs)):
    noise_ampl.append((fsr / 2) * 10 ** (noise_dbfs[x] / 20))

# Now build up the signal from the fundamental, harmonics, and noise tones
awf = np.zeros(npts)

for harmonic in range(len(harm_dbfs)):
    freq = fund_freq * (harmonic + 1)
    print("Frequency: ", freq)

    awf += gn.cos(npts, fs, harm_ampl[harmonic], freq, phase, td, tj)

# Maybe comment this and have FAEs uncomment to add noise?
for tone in range(len(noise_freqs)):
    freq = noise_freqs[tone]
    if gn.Window.NO_WINDOW == window:
        freq = gn.coherent(nfft, fs, noise_freqs[tone])
    print("Noise Frequency: ", freq)
    awf += gn.cos(npts, fs, noise_ampl[tone], freq, phase, td, tj)

# get quantizer noise in Volts
qnoise = 10 ** (qnoise_dbfs / 20)

ssb_fund = 4  # single side bin fundamental
ssb_rest = 5

# If we are not windowing then choose the closest coherent bin for fundamental
# Mark, you should check this out, with this logic the noise overlaps with the fundamental
if gn.Window.NO_WINDOW == window:
    freq = gn.coherent(nfft, fs, fund_freq)
    ssb_fund = 0
    ssb_rest = 0

# quantize waveform
qwf = gn.quantize(np.array(awf), fsr, qres, qnoise, code_fmt)

pl.figure(1)
pl.plot(awf[:10000])
# compute FFT
fft_cplx = gn.rfft(np.array(qwf), qres, navg, nfft, window, code_fmt, rfft_scale)
# compute frequency axis
freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs)
# compute FFT in db
fft_db = gn.db(fft_cplx)

# Fourier analysis configuration
key = 'fa'
gn.mgr_remove(key)
gn.fa_create(key)
gn.fa_analysis_band(key, "fdata*0.0", "fdata*1.0")
gn.fa_fixed_tone(key, 'A', gn.FaCompTag.SIGNAL, fund_freq, ssb_fund)
gn.fa_hd(key, 4)
gn.fa_ssb(key, gn.FaSsb.DEFAULT, ssb_rest)
gn.fa_ssb(key, gn.FaSsb.DC, -1)
gn.fa_ssb(key, gn.FaSsb.SIGNAL, -1)
gn.fa_ssb(key, gn.FaSsb.WO, -1)
gn.fa_fsample(key, fs)
print(gn.fa_preview(key, False))

# Fourier analysis results
fft_results = gn.fft_analysis(key, fft_cplx, nfft)
# compute THD
# result is in dBFS
fft_results['thd'] = (10 * np.log10(10 ** (fft_results['2A:mag_dbfs'] / 10) + (10 ** (fft_results['3A:mag_dbfs'] / 10))
                                    + (10 ** (fft_results['4A:mag_dbfs'] / 10))))

print("\nFourier Analysis Results:\n")
print("\nFrequency, Phase and Amplitude for Harmonics:\n")
for k in ['A:freq', 'A:mag_dbfs', 'A:phase',
          '2A:freq', '2A:mag_dbfs', '2A:phase',
          '3A:freq', '3A:mag_dbfs', '3A:phase',
          '4A:freq', '4A:mag_dbfs', '4A:phase',
          'wo:freq','wo:mag_dbfs', 'wo:phase']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))
print("\nFrequency, Phase and Amplitude for Noise:\n")
for k in ['wo:freq','wo:mag_dbfs', 'wo:phase']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))
print("\nSNR and THD \n")
for k in ['fsnr', 'thd']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))

# plot FFT
pl.figure(2)
fftax = pl.subplot2grid((1, 1), (0, 0), rowspan=2, colspan=2)
pl.title("FFT")
pl.plot(freq_axis, fft_db)
pl.grid(True)
pl.xlim(freq_axis[0], 20)
pl.ylim(-160.0, 20.0)
annots = gn.fa_annotations(fft_results)

for x, y, label in annots["labels"]:
    pl.annotate(label, xy=(x, y), ha='center', va='bottom')
for box in annots["tone_boxes"]:
    fftax.add_patch(MPRect((box[0], box[1]), box[2], box[3],
                           ec='pink', fc='pink', fill=True, hatch='x'))

pl.tight_layout()
pl.show()
