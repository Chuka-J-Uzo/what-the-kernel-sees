# Figures for "What the Kernel Sees". All values come from the original perf_stats
# and llama.cpp timing logs (Results.zip) and the measurement workbook.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Figure 1 ---------------------------------------------------------------
W = {'Mixtral-8x7B (Q2_K)': ([552,783,831,1061,527],[236.92,311.14,337.38,410.58,211.93]),
     'Llama-2-7B-Chat (Q5_K_M)': ([519,422,674,467,459],[190.22,179.16,233.09,163.26,151.74]),
     'Mistral-7B-Instruct (Q5_K_M)': ([315,153,351,324,292],[118.85,74.76,116.15,127.70,117.72])}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (m, (w, t)) in enumerate(W.items()):
    w, t = np.array(w), np.array(t)
    ax[0].scatter(w, t, s=28, color=f'C{i}', label=m, marker='os^'[i])
    s = stats.linregress(w, t); x = np.linspace(w.min(), w.max(), 10)
    ax[0].plot(x, s.intercept + s.slope * x, lw=1, color=f'C{i}')
ax[0].set_xlabel('Words generated in the run'); ax[0].set_ylabel('Wall-clock time (s)')
ax[0].set_title('(a) WasmEdge: five runs per model', fontsize=10); ax[0].legend(fontsize=7, frameon=False)

pre = np.array([226.2,215.6,242.3,218.0,241.1]); dec = np.array([2209.4,1904.0,2491.8,2973.9,4193.0])
wall = np.array([2506.50,2371.44,2799.82,3261.30,4506.83]); other = wall - pre - dec
cs = np.array([4.74,2.72,7.37,11.24,24.05])
x = np.arange(1, 6)
ax[1].bar(x, pre, color='C7', label='Prompt processing')
ax[1].bar(x, dec, bottom=pre, color='C0', label='Generation (6,530 tokens)')
ax[1].bar(x, other, bottom=pre+dec, color='C1', label='Other (load, exit)')
for xi, c, h in zip(x, cs, wall): ax[1].text(xi, h + 60, f'{c:.1f}M cs', ha='center', fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'run {i}' for i in x])
ax[1].set_ylabel('Time (s)'); ax[1].set_ylim(0, 5000)
ax[1].set_title('(b) Python, Mixtral-8x7B: identical output in every run', fontsize=10)
ax[1].legend(fontsize=7, frameon=False, loc='upper left')
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('fig_runs.pdf'); plt.savefig('fig_runs.png', dpi=130)

# Figure 2 ---------------------------------------------------------------
labels = ['Phi-2\n(same model)', 'Mistral-7B arch.\n(Python: Mistral;\nWasmEdge: Zephyr)']
py_pre, py_dec = [14.67, 6.75], [5.76, 3.35]
we_pre, we_dec = [244.2, 132.9], [9.90, 4.75]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, py, we, t in [(ax[0], py_pre, we_pre, '(a) Prompt processing'), (ax[1], py_dec, we_dec, '(b) Token generation')]:
    xx = np.arange(2)
    a.bar(xx - .18, py, .36, label='Python (CPU-only build)', color='C7')
    a.bar(xx + .18, we, .36, label='WasmEdge (CUDA build, 5-10 layers on GPU)', color='C0')
    for i in range(2): a.text(xx[i], max(py[i], we[i]) * 1.15, f'{we[i]/py[i]:.1f}x', ha='center', fontsize=9)
    a.set_xticks(xx); a.set_xticklabels(labels, fontsize=8); a.set_yscale('log'); a.set_ylabel('Tokens per second (log)')
    a.set_title(t, fontsize=10); a.grid(alpha=.3, axis='y'); a.set_ylim(1, 800)
ax[0].legend(fontsize=7, frameon=False, loc='upper right')
plt.tight_layout(); plt.savefig('fig_phases.pdf'); plt.savefig('fig_phases.png', dpi=130)

# Figure 3: GPU telemetry, first WasmEdge run of each model --------------------
import glob, re
import os
base = os.environ.get('LOGS', 'raw_logs') + '/RUST-WASM RESULTS/'  # run from the root of the data release
runs = [('Mixtral-8x7B', 'Mixtral*'), ('Llama-2-7B', 'LLAMA*'), ('Mistral-7B', 'Mistral*'),
        ('Zephyr-7B', 'Zephyr*'), ('Phi-2', 'Phi*')]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for i, (name, pat) in enumerate(runs):
    f = glob.glob(base + pat + '/gpu_stat (1).csv')[0]
    used, power = [], []
    for line in open(f, errors='ignore').read().splitlines()[1:]:
        p = line.split(',')
        if len(p) < 7: continue
        used.append(float(re.sub(r'[^\d.]', '', p[5]))); power.append(float(re.sub(r'[^\d.]', '', p[6])))
    t = np.arange(len(used))  # sampled once per second
    ax[0].plot(t, used, lw=1, color=f'C{i}', label=name)
    ax[1].plot(t, power, lw=.8, color=f'C{i}', label=name)
ax[0].axhline(6144, ls='--', lw=.8, color='k'); ax[0].text(5, 6144 * 0.96, '6 GB VRAM', fontsize=7, va='top')
ax[0].set_ylabel('GPU memory used (MiB)'); ax[1].set_ylabel('GPU power draw (W)')
ax[0].set_title('(a) GPU memory', fontsize=10); ax[1].set_title('(b) GPU power', fontsize=10)
for a in ax: a.set_xlabel('Seconds since logging started (approx.)'); a.grid(alpha=.3)
ax[0].legend(fontsize=7, frameon=False, loc='center right')
plt.tight_layout(); plt.savefig('fig_gpu.pdf'); plt.savefig('fig_gpu.png', dpi=130)
