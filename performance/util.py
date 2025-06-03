import numpy as np

def get_mean_std_iqr_label(array, precision=2):
    mean = array.mean()
    std = array.std()
    iqr = np.subtract(*np.percentile(array, [75, 25]))

    if precision == 1:
        label = f'$\mu$: {mean:.1f} $\sigma$: {std:.1f} IQR: {iqr:.1f}'
    elif precision == 2:
        label = f'$\mu$: {mean:.2f} $\sigma$: {std:.2f} IQR: {iqr:.2f}'
    elif precision == 3:
        label = f'$\mu$: {mean:.3f} $\sigma$: {std:.3f} IQR: {iqr:.3f}'

    return label, (mean, std, iqr)