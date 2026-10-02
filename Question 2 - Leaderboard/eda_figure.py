"""Task 2 exploratory figure: series excerpt, ACF and periodogram of the training target."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

y = pd.read_csv("Data/student_train.csv").value.values.astype(float)
n = len(y)
yc = y - y.mean()
spec = np.fft.rfft(yc, n=2 * n)
acf = np.fft.irfft(spec * np.conj(spec))[:n]
acf /= acf[0]
power = np.abs(np.fft.rfft(yc)) ** 2
freq = np.fft.rfftfreq(n)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].plot(np.arange(n - 1000, n), y[-1000:], lw=0.8)
ax[0].set(title="Last 1000 observed steps", xlabel="time_idx", ylabel="value")
ax[1].stem(np.arange(0, 200), acf[:200], markerfmt=" ", basefmt=" ")
for k in (24, 48, 72, 168):
    ax[1].axvline(k, color="grey", ls=":", lw=0.8)
ax[1].set(title="Autocorrelation (lags 0-199)", xlabel="lag", ylabel="ACF")
mask = (freq > 1 / 400) & (freq < 0.5)
ax[2].semilogy(1 / freq[mask], power[mask], lw=0.6)
ax[2].axvline(24, color="red", ls=":", lw=1, label="period 24")
ax[2].axvline(12, color="orange", ls=":", lw=1, label="period 12")
ax[2].set(xscale="log", title="Periodogram (periods 2-400)", xlabel="period (steps)", ylabel="power")
ax[2].legend()
fig.tight_layout()
fig.savefig("../report/figures/t2_eda.pdf")
print("quantiles 50/90/99/max:", np.quantile(y, [.5, .9, .99, 1]).round(1))
print("acf at 1,12,24,48,96,168:", acf[[1, 12, 24, 48, 96, 168]].round(3))
