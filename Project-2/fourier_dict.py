import numpy as np
import matplotlib.pyplot as plt

def create_fourier_dictionary(N, M=None):
    """
    Create a complex Fourier dictionary of size N x M.
    If M is None, use M = N (complete dictionary).
    """
    if M is None:
        M = N
    n = np.arange(N)
    k = np.arange(M)
    D = np.exp(2j * np.pi * np.outer(n, k) / N)  # complex exponentials
    D /= np.sqrt(N)  # normalize
    return D

# Parameters
N = 1024  # frame length
M = N     # dictionary size

# Create the dictionary
D = create_fourier_dictionary(N, M)

# Plot the real and imaginary parts of the first 6 atoms
num_atoms_to_plot = 6
plt.figure(figsize=(14, 8))
for i in range(num_atoms_to_plot):
    plt.subplot(2, num_atoms_to_plot, i+1)
    plt.plot(np.real(D[:, i]))
    plt.title(f'Real Part - Atom {i}')
    plt.xlabel('Sample Index')

    plt.subplot(2, num_atoms_to_plot, num_atoms_to_plot + i + 1)
    plt.plot(np.imag(D[:, i]))
    plt.title(f'Imag Part - Atom {i}')
    plt.xlabel('Sample Index')

plt.tight_layout()
plt.suptitle('Fourier Dictionary Atoms - Real and Imaginary Parts', y=1.02, fontsize=16)
plt.show()
# The Fourier dictionary is a set of complex exponentials that can be used for various signal processing tasks.
