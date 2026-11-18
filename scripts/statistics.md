---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: HPMC local environment
    language: python
    name: .venv
---

```python
from pathlib import Path
import numpy as np

def load_data(folder: str, plot_from: int = 20000) -> tuple[np.ndarray, np.ndarray, int]:
    folders = sorted([p.name for p in Path(folder).iterdir() if p.is_dir()])

    all_y = []
    steps = None
    
    for f in folders:
        with np.load(f"{folder}/{f}/results.npz") as data:
            y = data["delta_F_hist"][plot_from:]
            all_y.append(y)
            
            batches = data["batches_executed"]
            steps = np.arange(plot_from * 1000, batches * 1000, 1000)

    all_y = np.vstack(all_y)
    n_seeds = all_y.shape[0]

    return steps, all_y, n_seeds

def get_final_stats(folder: str) -> tuple[float, float]:
    folders = sorted([p.name for p in Path(folder).iterdir() if p.is_dir()])

    n = len(folders)

    y = np.zeros(n)

    for i, f in enumerate(folders):
        with np.load(f"{folder}/{f}/results.npz") as data:
            y[i] = data["delta_F"]

    mean_y = np.mean(y)
    sem_y = np.std(y, ddof=1) / np.sqrt(n)

    return mean_y, sem_y
```

```python
mean_2h, sem_2h = get_final_stats("../3c2h_6x6x12")
mean_4h, sem_4h = get_final_stats("../3c4h_6x6x12")
mean_6h1, sem_6h1 = get_final_stats("../3c6h1_6x6x12")
mean_6h2, sem_6h2 = get_final_stats("../3c6h2_6x6x12")
mean_9r, sem_9r = get_final_stats("../3c6h2_6x6x12")

A = np.array([
    [2.0, 0.0, 0.0, 2.0],
    [1.0, 2.0, 0.0, 1.0],
    [2/3, 4/3, 4/3, 2.0],
    [4/3, 4/3, 2/3, 2/3],
])

y = np.array([mean_2h, mean_4h, mean_6h1, mean_6h2])
err = np.array([sem_2h, sem_4h, sem_6h1, sem_6h2])

A_inv = np.linalg.inv(A)
params = A_inv @ y
cov = A_inv @ np.diag(err ** 2) @ A_inv
param_err = np.sqrt(np.diag(cov))

for name, val, u in  zip(["h", "J", "J'", "h'"], params, param_err):
    print(f"{name:3s} = {val:+.6f} +/- {u:.6f}")

predict_mean_9r = params.dot([4/3, 4/3, 4/3, 0])
predict_sem_9r = param_err.dot([4/3, 4/3, 4/3, 0])
print(predict_mean_9r, predict_sem_9r)
print(mean_9r, sem_9r)
```

```python
import re
import matplotlib.pyplot as plt

outdir = "../3c4h_6x6x12"
plot_from = 20000

x, y, n_seeds = load_data(outdir, plot_from)

n_steps = len(x)
mean_y = np.mean(y, axis=0)
sem_y = np.std(y, axis=0, ddof=1) / np.sqrt(n_seeds)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y.T, color="gray", alpha=0.25, linewidth=1.0)
ax.plot(x, mean_y, color="tab:blue", linewidth=2.0, label=f"Mean ({n_seeds} independent seeds)")
ax.fill_between(x, mean_y - sem_y, mean_y + sem_y, color="tab:blue", alpha=0.2, label="±1 SEM (shaded)")

stride = max(1, (n_steps - burn_in_idx) // 20)  # ~20 error bars across the plot

ax.errorbar(
    x[::stride],
    mean_y[::stride],
    yerr=sem_y[::stride],
    fmt="o",
    color="navy",
    ecolor="navy",
    markersize=4,
    elinewidth=1.3,
    capsize=3,
    label="±1 SEM (sampled)",
    zorder=5
)

ax.set_title(f"LSMC Convergence (Final: {mean_y[-1]:.5f} ± {sem_y[-1]:.5f})")
ax.set_xlabel("MC Steps")
ax.set_ylabel(r"$\Delta F / N k_B T$")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")

plt.tight_layout()
plt.show()
```

```python

```
