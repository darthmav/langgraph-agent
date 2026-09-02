# Spectral Signal Processing

## Introduction

Spectral signal processing analyzes signals in the frequency domain rather than the time domain. The core idea is that any signal can be decomposed into a sum of sinusoidal components at different frequencies.

## Fourier Transform

### Continuous Fourier Transform

$$X(f) = \int_{-\infty}^{\infty} x(t) e^{-2\pi i f t} dt$$

### Discrete Fourier Transform (DFT)

For a sequence $x[n]$ of length $N$:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-2\pi i kn/N}, \quad k = 0, 1, \ldots, N-1$$

### Fast Fourier Transform (FFT)

Efficient algorithm to compute DFT:
- Complexity: $O(N \log N)$ vs $O(N^2)$ for direct DFT
- Implemented in `numpy.fft` and `scipy.fft`

## Key Concepts

### Frequency Spectrum

The magnitude $|X[k]|$ shows the strength of each frequency component.

### Phase Spectrum

The angle $\angle X[k]$ shows the phase shift of each component.

### Nyquist Frequency

Maximum representable frequency: $f_{Nyquist} = f_s / 2$

where $f_s$ is the sampling frequency.

## Spectral Filtering

### Low-Pass Filter

Allows low frequencies, attenuates high frequencies:

$$H_{LP}[k] = \begin{cases} 1 & |k| < k_c \\ 0 & \text{otherwise} \end{cases}$$

### High-Pass Filter

Allows high frequencies, attenuates low frequencies:

$$H_{HP}[k] = \begin{cases} 0 & |k| < k_c \\ 1 & \text{otherwise} \end{cases}$$

### Band-Pass Filter

Allows frequencies in a specific range.

## Filtering Process

1. Compute FFT of signal: $X = \text{FFT}(x)$
2. Multiply by filter: $Y[k] = H[k] \cdot X[k]$
3. Compute inverse FFT: $y = \text{IFFT}(Y)$

## Applications

- **Audio Processing**: Equalization, noise reduction
- **Image Processing**: Blurring, edge detection
- **Communications**: Modulation, demodulation
- **Feature Extraction**: Frequency-domain features for ML

## Python Implementation

```python
import numpy as np
from scipy import fft

# Forward FFT
X = fft.fft(x)

# Inverse FFT
x_reconstructed = fft.ifft(X)

# Frequency axis
freqs = fft.fftfreq(len(x), d=1/fs)
```

## Numerical Considerations

- FFT requires power-of-2 lengths for optimal performance (not strictly required)
- Real signals have conjugate-symmetric spectra
- Use `rfft` for real-valued signals (more efficient)
- Window functions reduce spectral leakage

## References

1. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing*. Pearson.
2. Proakis, J. G., & Manolakis, D. G. (2007). *Digital Signal Processing*. Pearson.
3. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*. California Technical Publishing.
