import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

out_dir = 'Results Collective Displacement/'

# Energy Variation Plot

plt.figure(dpi=150)
ax = plt.gca()

stats = np.loadtxt(out_dir + "stats.txt", dtype=float)
X = stats[0]
Y = stats[3]
ax.plot(X, Y, label=out_dir)

plt.legend()
ax.set_title('Maximum change of energy')
plt.savefig(out_dir + 'Energy variation (sweep).pdf')

# Expectation values Plots

p = PdfPages(out_dir + 'Expectation Values.pdf')
figs = []
stepsize = 1

# Spin Density
expectation_sigmaz = np.loadtxt(out_dir + "SpinDensity.txt", dtype=float)
X = np.arange(0, len(expectation_sigmaz))

fig1 = plt.figure(dpi=150)
plt.plot(X, expectation_sigmaz, '-o')
plt.title('Spin Density $<\dfrac{1+\sigma^z}{2}>$')
plt.xlabel('Spin index i')
plt.ylabel('$<\dfrac{1+\sigma^z}{2}>$')

ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(np.trunc(start), np.trunc(end)+1, stepsize))

figs.append(fig1)

# Phonon
phonon = np.loadtxt(out_dir + "Phonon.txt", dtype=float)
X = np.arange(0, len(phonon))

fig2 = plt.figure(dpi=150)
plt.plot(X, phonon, '-o')
plt.title('Phonon Number')
plt.xlabel('Spin index i')
plt.ylabel('$<b_i^{\dagger} b_i>$')

ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(np.trunc(start), np.trunc(end)+1, stepsize))

figs.append(fig2)

# Four-Point Spin Correlator
O_SC = np.loadtxt(out_dir + "O_SC.txt", dtype=float)

O_SC_delta = []
for osc in O_SC:
    O_SC_delta.append(np.mean(osc))

X = np.arange(0, len(O_SC_delta))
fig3 = plt.figure(dpi=150)
plt.plot(X, O_SC_delta, '-o')
plt.title('Four-Point Spin Correlator as a function of delta $<\sigma_i^+ \sigma_{i+1}^+ \sigma_{i+\delta}^- \sigma_{i+1+\delta}^-> (\delta)$')
plt.xlabel('$\delta$')
plt.ylabel('$<\sigma_i^+ \sigma_{i+1}^+ \sigma_{i+\delta}^- \sigma_{i+1+\delta}^->$')

ax = plt.gca()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(np.trunc(start), np.trunc(end)+1, stepsize))

figs.append(fig3)

# Spin-Spin
spinSpin = np.loadtxt(out_dir + "SpinSpin.txt", dtype=float)

fig4 = plt.figure(dpi=150)
ax = plt.gca()

im = ax.imshow(spinSpin)
fig4.colorbar(im, ax = ax)

ax.set_title('Spin-Spin $<\sigma_i^z \sigma_j^z>$')
ax.set_xlabel('Spin index i')
ax.set_ylabel('Spin index j')

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(np.trunc(start), np.trunc(end)+1, stepsize))

figs.append(fig4)

# Spin-Phonon Correlator
# /!\ important to be in the local picture, otherwise the following code doesn't give what we want
Pi = np.loadtxt(out_dir + "SpinPhononCorrelator.txt", dtype=float)

fig5 = plt.figure(dpi=150)
ax = plt.gca()

im = ax.imshow(Pi, cmap='magma')
fig5.colorbar(im, ax = ax)

ax.set_title('Spin-Phonon Correlator $<\sigma_i^z r_j> - <\sigma_i^z> <r_j>$')
ax.set_xlabel('Phonon index j')
ax.set_ylabel('Spin index i')

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(np.trunc(start), np.trunc(end)+1, stepsize))

figs.append(fig5)


for fig in figs:
    fig.savefig(p, format='pdf')
p.close()