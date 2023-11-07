import libm2k
import math
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

max_buffer_size = 500000


pg_available_sample_rates = [1000, 10000, 100000, 1000000, 10000000, 100000000]
pg_max_rate = pg_available_sample_rates[-1]  # last sample rate = max rate
pg_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

min_nr_of_points = 10


def get_best_ratio(ratio):
    max_it = max_buffer_size / ratio
    best_ratio = ratio
    best_fract = 1

    for i in range(1, int(max_it)):
        new_ratio = i * ratio
        (new_fract, integral) = math.modf(new_ratio)
        if new_fract < best_fract:
            best_fract = new_fract
            best_ratio = new_ratio
        if new_fract == 0:
            break

    return best_ratio, best_fract


def get_samples_count(rate, freq):
    ratio = rate / freq
    if ratio < min_nr_of_points and rate < pg_max_rate:
        return 0
    if ratio < 2:
        return 0

    ratio, fract = get_best_ratio(ratio)
    # ratio = number of periods in buffer
    # fract = what is left over - error

    size = int(ratio)
    while size & 0x03:
        size = size << 1
    while size < 1024:
        size = size << 1
    return size


def get_optimal_sample_rate_pg(freq):
    for rate in pg_available_sample_rates:
        buf_size = get_samples_count(rate, freq)
    if buf_size:
        return rate


def square_buffer_generator(freq, phase, sample_rate, dutycycle=0.5):
    buffer = []

    nr_of_samples = get_samples_count(sample_rate, freq)
    samples_per_period = sample_rate / freq
    phase_in_samples = ((phase / 360) * samples_per_period)
    scaler = freq / sample_rate
    shift: float = dutycycle / 2
    for i in range(nr_of_samples):
        val = 0 if (((i + phase_in_samples) * scaler + shift) % 1 < dutycycle) else 1
        buffer.append(val)

    return buffer


def square_wave_digital(sig, channel):
    # shifts buffer to the corresponding DIO channel
    dig_buf = list(map(lambda s: int(s) << channel, sig))
    for i in range(8):
        dig_buf.extend(dig_buf)
    return dig_buf


def lcm(x, y, z):
    gcd2 = math.gcd(y, z)
    gcd3 = math.gcd(x, gcd2)

    lcm2 = y * z // gcd2
    lcm3 = x * lcm2 // math.gcd(x, lcm2)
    return int(lcm3)


def extend_buffer(buf, desired_length):

    times = int(desired_length / len(buf))
    aux = copy.deepcopy(buf)

    for i in range(1, times):
        buf.extend(aux)
    return buf


def FFT(t, y):
    n = len(t)
    delta = (max(t) - min(t)) / (n-1)
    k = int(n/2)
    f = np.arange(k) / (n* delta)
    Y = abs(np.fft.fft(y))[:k]
    return f, Y


def lowpass(x, alpha=0.01):
    data = [x[0]]
    for a in x[1:]:
        data.append(data[-1] + (alpha*(a-data[-1])))
    return np.array(data)

def main():
    uri = "ip:192.168.2.1"

    # Connect to M2K and Initialize ADC and Pattern Generator Objects
    ctx = libm2k.m2kOpen(uri)
    ctx.calibrateADC()
    adc = ctx.getAnalogIn()
    digital = ctx.getDigital()
    ps = ctx.getPowerSupply()

    # enable and set power supply pins to +5, -5 V to power up OP Amp
    ps.reset()
    ps.enableChannel(0, True)
    ps.pushChannel(0, 5)
    ps.enableChannel(1, True)
    ps.pushChannel(1, -5)

    # Create 3 digital clock pattern buffers at 3 different frequencies
    # We will use these to drive the RGB LED
    # Each signal will turn the LED either Red, Blue or Green

    red_freq = 500  # in Hz
    green_freq = 600  # in Hz
    blue_freq = 700  # in Hz

    # Do we want to play around with phase, offset and duty cycle ?
    square_dutycycle = 0.5
    square_offset = 0
    square_phase = 0

    sig = square_buffer_generator(red_freq, square_phase, pg_available_sample_rates[1], square_dutycycle)
    red_buf = square_wave_digital(sig, pg_channels[13])

    sig = square_buffer_generator(green_freq, square_phase, pg_available_sample_rates[1], square_dutycycle)
    green_buf = square_wave_digital(sig, pg_channels[14])

    sig = square_buffer_generator(blue_freq, square_phase, pg_available_sample_rates[1], square_dutycycle)
    blue_buf = square_wave_digital(sig, pg_channels[15])

    # Make sure all buffers are the same length
    # Calculate The Least Common Multiple(LCM) between the lengths of the 3 buffers
    # Extend each buffer until they are the length of LCM

    buffer_length = lcm(len(red_buf), len(green_buf), len(blue_buf))
    red_buf = extend_buffer(red_buf, buffer_length)
    blue_buf = extend_buffer(blue_buf, buffer_length)
    green_buf = extend_buffer(green_buf, buffer_length)

    # Find and set optimal sample rate for the 3 buffers, using LCM
    digital.setSampleRateOut(pg_available_sample_rates[1])

    # Enable and configure M2K Digital pins
    # For our board this is range(13,16)
    for i in range(13, 16):
        digital.setDirection(i, libm2k.DIO_OUTPUT)
        digital.enableChannel(i, True)
    digital.setCyclic(True)

    buffer = []
    if len(red_buf) == len(blue_buf) == len(green_buf):
        for i in range(len(red_buf)):
            bit = red_buf[i] + blue_buf[i] + green_buf[i]
            buffer.append(bit)
    digital.push(buffer)

    # Configure Analog Inputs
    adc.enableChannel(0, True)
    adc.enableChannel(1, True)

    adc.setSampleRate(pg_available_sample_rates[1])
    adc.setRange(0, -1, 1)
    adc.setRange(1, -1, 1)
    # Get data from M2K
    data = adc.getSamples(pow(2, 10))  # power of 2 samples
    measured_data = data[1]
    reference_data = data[0]

    # Fun Stuff
    balanced_signal = measured_data - np.mean(measured_data)
    windowed_signal = balanced_signal * np.blackman(len(balanced_signal))
    #windowed_signal = lowpass(windowed_signal)
    measured_data_fft = np.fft.fft(windowed_signal)
    # d - time step, inverse of the sampling rate
    freq = np.fft.fftfreq(len(balanced_signal), d=1/pg_available_sample_rates[1])

    # we do fft shift because fft function plots positive frequencies first
    # without fft shift we get a line connecting the last point of the positive
    # frequencies to the first point of the negative frequencies.
    measured_data_fft = np.fft.fftshift(measured_data_fft)
    freq = np.fft.fftshift(freq)
    N = len(balanced_signal)

    # now for the reference data
    balanced_ref = reference_data - np.mean(reference_data)
    windowed_ref = balanced_ref * np.blackman(len(balanced_ref))
    ref_data_fft = np.fft.fft(windowed_ref)
    # d - time step, inverse of the sampling rate
    ref_freq = np.fft.fftfreq(len(balanced_ref), d=1/pg_available_sample_rates[1])
    ref_data_fft = np.fft.fftshift(ref_data_fft)
    ref_freq = np.fft.fftshift(ref_freq)

    plt.plot(ref_freq[N // 2:], 2.0 / N * np.abs(ref_data_fft[N // 2:]))
    plt.plot(freq[N // 2:], 2.0 / N * np.abs(measured_data_fft[N // 2:]))
    plt.show()

    #500 , 600 , 700 Hz
    libm2k.contextClose(ctx)


main()
