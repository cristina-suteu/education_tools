import libm2k
import numpy as np
import matplotlib.pyplot as plt
import colorimeter_functions


uri = "ip:192.168.2.1"

# Connect to M2K and Initialize ADC and Pattern Generator Objects
ctx = libm2k.m2kOpen(uri)
ctx.calibrateADC()
adc = ctx.getAnalogIn()
digital = ctx.getDigital()
ps = ctx.getPowerSupply()

# Enable and set power supply pins to +5, -5 V to power up OP Amp
ps.reset()
ps.enableChannel(0, True)
ps.pushChannel(0, 5)
ps.enableChannel(1, True)
ps.pushChannel(1, -5)


# Configure Analog Inputs

adc.enableChannel(0, True)
adc.enableChannel(1, True)

adc.setSampleRate(colorimeter_functions.pg_available_sample_rates[1])
adc.setRange(0, -1, 1)
adc.setRange(1, -1, 1)

digital.setSampleRateOut(colorimeter_functions.pg_available_sample_rates[1])

# Enable and configure M2K Digital pins as outputs
# For our add-on board, we need to configure DIO13,DIO14,DIO15
for i in range(13, 16):
    digital.setDirection(i, libm2k.DIO_OUTPUT)
    digital.enableChannel(i, True)
digital.setCyclic(True)

# Create digital buffer, to be pushed to Digital Outputs
# this is used to drive the RGB LED

digital_buffer = colorimeter_functions.create_digital_buffer()
digital.push(digital_buffer)

# Create figure to plot results
fig, (ax1, ax2) = plt.subplots(nrows=2)
fig.set_figheight(6)
fig.set_figwidth(6)
# Where the magic happens

while True:
    # Get data from M2K

    ref_data = adc.getSamples(pow(2, 10))[0]
    measured_data = adc.getSamples(pow(2, 10))[1]

    # Calculate FFT and Light Absorbance
    ref_freq, ref_data_fft, ref_length = colorimeter_functions.compute_fft(ref_data)
    freq, measured_data_fft, length = colorimeter_functions.compute_fft(measured_data)
    red_abs, green_abs, blue_abs = colorimeter_functions.light_absorbance(
        measured_data_fft, ref_data_fft, freq, length)

    # plot FFT and absorbance
    ax1.clear()
    ax1.clear()
    ax1.set_title("FFT Plot")
    ax2.set_title("Absorbance Plot")
    ax1.plot(freq[length // 2:], 2.0 / length * np.abs(ref_data_fft[length // 2:]))
    ax1.plot(freq[length // 2:], 2.0 / length * np.abs(measured_data_fft[length // 2:]))
    bar_colors = ['tab:red', 'tab:green', 'tab:blue']
    colors = ['red', 'green', 'blue']
    absorbance = [red_abs, green_abs, blue_abs]
    ax2.bar(colors, absorbance, color=bar_colors)
    ax2.set_ylim(0, 100)
    plt.show(block=False)
    plt.pause(5)
    if KeyboardInterrupt:
        libm2k.contextClose(ctx)
        break




