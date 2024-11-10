# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr

usetex = False

if not usetex:
    # computer modern math text
    mpl.rcParams.update({'mathtext.fontset': 'cm'})
mpl.rc('font', **{'family': 'serif',
       'serif': ['Times' if usetex else 'Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=usetex)
# %%
def sign_floor(x):
    return np.sign(x) * np.floor(np.abs(x))

def sign_ceil(x):
    return np.sign(x) * np.ceil(np.abs(x))
# %%
alpha = 0
den = 100
p = 1e6/den
real_ord = 1
real_lam = 630
beta_0 = np.rad2deg(np.arcsin(real_ord * real_lam / p - np.sin(np.deg2rad(alpha))))
beta_1 = np.rad2deg(np.arcsin(2 * real_lam / p + np.sin(np.deg2rad(alpha))))
# %%

beta = np.linspace(-50, 50, 2000)
dbeta = np.mean(np.diff(beta))

prod = np.sin(np.deg2rad(alpha) + np.deg2rad(beta)) * p
# %%
lam_min = 610
lam_max = 650

n1_min, n1_max = sign_ceil(np.min(prod / lam_min)), sign_floor(np.max(prod / lam_min))
n2_min, n2_max = sign_ceil(np.min(prod / lam_max)), sign_floor(np.max(prod / lam_max))

n_min = int(min(n1_min, n2_min))
n_max = int(max(n1_max, n2_max))

for n in range(n_min, n_max + 1):
    if n == 0:
        continue
    lam = prod / n
    lam[np.where((lam < lam_min) | (lam > lam_max))] = np.nan
    plt.plot(beta, lam, label=f'n={n}')
# plt.legend()

# %%
