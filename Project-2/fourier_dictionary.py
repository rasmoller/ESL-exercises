import numpy as np
import matplotlib.pyplot as plt


class FourierDictionary:
    def __init__(self, N=1024, M=None):
        self.N = N
        self.M = M

    def create_fourier_dictionary(self):
        """
        Create a real Fourier dictionary of size N x M.
        If M is None, use M = N (complete dictionary).
        """
        if self.M is None:
            self.M = self.N
        n = np.arange(self.N)
        k = np.arange(self.M)
        D = np.exp(2j * np.pi * np.outer(n, k) / self.N) 
        D /= np.sqrt(self.N) 

        return np.abs(D)


# Example usage
if __name__ == "__main__":
    # Create the dictionary
    D = FourierDictionary().create_fourier_dictionary()

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
