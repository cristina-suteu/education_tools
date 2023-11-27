import genalyzer
import genalyzer_advanced as gn
import numpy as np
import matplotlib.pyplot as plt

# Waveform Generation Parameters
ttype = 0
npts = 16384
sample_rate = 1000.0 # 900000000.0#
num_tones = 2
tone_freq = [1.0 , 2.0]# 250000.0
tone_ampl = [10, 10]
tone_phase = [0, 0]

# Quantizer parameters
fsr = 3.0
qres = 16
qnoise_dbfs = 0.0#-63.0

# Generate Single Harmonic Tone
c = genalyzer.config_gen_tone(ttype, npts, sample_rate, num_tones, tone_freq, tone_ampl, tone_phase)
awf1 = genalyzer.gen_real_tone(c)
plt.plot(awf1)
#plt.show()
# Quantize Waveform
qnoise = 10**(qnoise_dbfs / 20)
genalyzer.config_quantize(npts, fsr, qres, qnoise, c)
qwf1 = genalyzer.quantize(awf1, c)
plt.plot(qwf1)
#plt.show()

# Fourier Analysis
inputs = dict()
inputs["domain_wf"] = 0
inputs["type_wf"] = 1
inputs["nfft"] = len(qwf1)
inputs["npts"] = len(qwf1)
inputs["navg"] = 1
inputs["fs"] = int(sample_rate)
inputs["fsr"] = fsr
inputs["qres"] = qres
inputs["win"] = gn.Window.NO_WINDOW


c = genalyzer.config_fftz(inputs['npts'], inputs['qres'], inputs['navg'], inputs['nfft'], inputs['win'])
genalyzer.config_set_sample_rate(inputs['fs'], c)

# Find tones
genalyzer.config_fa(tone_freq[0], c)
#genalyzer.gn_config_fa_auto(ssb_width=120, c=c)
# Compute FFT
qwf1_q = [0]
fft_out_i, fft_out_q = genalyzer.fftz(qwf1, qwf1_q , c)

# Get all Fourier analysis results
all_results = genalyzer.get_fa_results(fft_out_i, c)
print(all_results)
# Plot FFT

fft_cp = np.roll(np.absolute(fft_out_i), round(0.5*inputs['nfft']))
f = np.linspace(start=-0.5*inputs['fs'], stop=0.5*inputs['fs'], num=inputs['nfft'])
dbfs_data = 10*np.log10(np.square(fft_cp))

# compute FSNR and THD

snr = genalyzer.get_fa_single_result("snr", fft_out_i, c)
thd_dbfs = np.log10(10 ** (all_results['2A:mag_dbc']/10)) + all_results["A:mag_dbfs"]

"""+
               10 ** (all_results['3A:mag_dbc']/10) +
               10 ** (all_results['4A:mag_dbc']/10) +
               10 ** (all_results['5A:mag_dbc']/10))"""
print("SNR " + str(snr) + "\n")
print("THD " + str(thd_dbfs) + "\n")


plt.clf()
plt.plot(f[8092:8292], dbfs_data[8092:8292])


# Add markers for the harmonics
harmonic_keys = ['A','2A','dc']

for key in harmonic_keys:
    freq = all_results[f'{key}:freq']
    print(freq)
    amp = all_results[f'{key}:mag_dbfs']
    print(amp)
    plt.plot(freq, amp, 'ro')


plt.ylim([-140, 0])
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [dBFs]")
plt.draw()
plt.show()
