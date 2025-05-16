from cr.sparse.dict import fourier_basis

import numpy as np
import matplotlib.pyplot as plt



fDict = fourier_basis(10)

print(fDict.shape)
print(fDict.size)

# Plot the Fourier basis vectors
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    ax.plot(fDict[:, i])
    ax.set_title(f'Basis Vector {i+1}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

plt.tight_layout()
plt.show()
