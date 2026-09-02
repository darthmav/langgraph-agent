"""
Fourier Analysis Example

Demonstrates basic FFT analysis and visualization of signals.
Shows how to decompose signals into frequency components.

References:
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing.
- Smith, S. W. (1997). The Scientist and Engineer's Guide to DSP.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft


def generate_test_signal(fs=1000, duration=1.0):
    """
    Generate a test signal with multiple frequency components.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    duration : float
        Signal duration in seconds
    
    Returns:
    --------
    t : ndarray
        Time array
    signal : ndarray
        Composite signal
    components : dict
        Individual signal components
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Component 1: 50 Hz sine wave
    f1, a1 = 50, 1.0
    comp1 = a1 * np.sin(2 * np.pi * f1 * t)
    
    # Component 2: 120 Hz sine wave
    f2, a2 = 120, 0.5
    comp2 = a2 * np.sin(2 * np.pi * f2 * t)
    
    # Component 3: 200 Hz sine wave
    f3, a3 = 200, 0.3
    comp3 = a3 * np.sin(2 * np.pi * f3 * t)
    
    # Composite signal
    signal = comp1 + comp2 + comp3
    
    # Add some noise
    noise = 0.2 * np.random.randn(len(t))
    signal_noisy = signal + noise
    
    components = {
        '50Hz': comp1,
        '120Hz': comp2,
        '200Hz': comp3,
        'noise': noise,
        'clean': signal,
        'noisy': signal_noisy
    }
    
    return t, signal, signal_noisy, components


def compute_fft(signal, fs):
    """
    Compute FFT and return frequency spectrum.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    fs : float
        Sampling frequency
    
    Returns:
    --------
    freqs : ndarray
        Frequency array (positive frequencies only)
    magnitude : ndarray
        Magnitude spectrum
    phase : ndarray
        Phase spectrum
    """
    n = len(signal)
    
    # Compute FFT
    fft_result = fft.fft(signal)
    
    # Frequency array
    freqs = fft.fftfreq(n, d=1/fs)
    
    # Take only positive frequencies
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    fft_result = fft_result[positive_mask]
    
    # Compute magnitude and phase
    magnitude = np.abs(fft_result) * 2 / n  # Normalize
    phase = np.angle(fft_result)
    
    # DC component doesn't need factor of 2
    magnitude[0] = magnitude[0] / 2
    
    return freqs, magnitude, phase


def plot_time_domain(t, signal, signal_noisy, components):
    """
    Plot signals in time domain.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot individual components
    axes[0].plot(t, components['50Hz'], label='50 Hz', alpha=0.7)
    axes[0].plot(t, components['120Hz'], label='120 Hz', alpha=0.7)
    axes[0].plot(t, components['200Hz'], label='200 Hz', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Individual Frequency Components')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 0.1)  # Show first 100ms for clarity
    
    # Plot clean composite signal
    axes[1].plot(t, components['clean'], color='steelblue')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Clean Composite Signal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 0.1)
    
    # Plot noisy signal
    axes[2].plot(t, components['noisy'], color='coral', label='Noisy', alpha=0.7)
    axes[2].plot(t, components['clean'], color='steelblue', label='Clean', alpha=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Noisy Composite Signal')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 0.1)
    
    plt.tight_layout()
    return fig


def plot_frequency_spectrum(freqs, magnitude_clean, magnitude_noisy):
    """
    Plot frequency spectrum.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Clean signal spectrum
    axes[0].stem(freqs, magnitude_clean, linefmt='steelblue', 
                 markerfmt='bo', basefmt='k-')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Frequency Spectrum (Clean Signal)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 300)
    
    # Noisy signal spectrum
    axes[1].stem(freqs, magnitude_noisy, linefmt='coral', 
                 markerfmt='ro', basefmt='k-')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Spectrum (Noisy Signal)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 300)
    
    plt.tight_layout()
    return fig


def plot_phase_spectrum(freqs, phase):
    """
    Plot phase spectrum.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Only plot phase for significant magnitudes
    ax.scatter(freqs, phase, s=10, alpha=0.5, color='purple')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Phase Spectrum')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    
    plt.tight_layout()
    return fig


def demonstrate_frequency_resolution(fs, duration):
    """
    Demonstrate frequency resolution dependence on signal duration.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Two close frequencies
    f1, f2 = 50, 55
    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    freqs, magnitude, _ = compute_fft(signal, fs)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, magnitude, color='green')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Frequency Resolution Demo (T = {duration}s, Δf = {1/duration:.2f} Hz)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 65)
    
    plt.tight_layout()
    return fig


def main():
    """Main demonstration of Fourier analysis."""
    print("=" * 60)
    print("Fourier Analysis Demonstration")
    print("=" * 60)
    
    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    duration = 1.0  # Signal duration (seconds)
    
    print(f"\nSignal parameters:")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Number of samples: {fs * duration}")
    print(f"  Nyquist frequency: {fs / 2} Hz")
    print(f"  Frequency resolution: {1 / duration} Hz")
    
    # Generate signal
    print("\nGenerating test signal with components at 50 Hz, 120 Hz, and 200 Hz...")
    t, signal, signal_noisy, components = generate_test_signal(fs, duration)
    
    # Compute FFT
    print("Computing FFT...")
    freqs, mag_clean, phase_clean = compute_fft(components['clean'], fs)
    freqs, mag_noisy, phase_noisy = compute_fft(components['noisy'], fs)
    
    # Find peaks in spectrum
    print("\nDetected frequency peaks (noisy signal):")
    peak_threshold = 0.1 * np.max(mag_noisy)
    peak_indices = np.where(mag_noisy > peak_threshold)[0]
    
    # Group nearby peaks
    detected_freqs = []
    if len(peak_indices) > 0:
        current_group = [freqs[peak_indices[0]]]
        for i in range(1, len(peak_indices)):
            if freqs[peak_indices[i]] - freqs[peak_indices[i-1]] < 10:
                current_group.append(freqs[peak_indices[i]])
            else:
                detected_freqs.append(np.mean(current_group))
                current_group = [freqs[peak_indices[i]]]
        detected_freqs.append(np.mean(current_group))
    
    for freq in detected_freqs:
        print(f"  {freq:.1f} Hz")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    fig1 = plot_time_domain(t, signal, signal_noisy, components)
    plt.savefig('examples/fourier_time_domain.png', dpi=150, bbox_inches='tight')
    print("Saved time domain plot to examples/fourier_time_domain.png")
    
    fig2 = plot_frequency_spectrum(freqs, mag_clean, mag_noisy)
    plt.savefig('examples/fourier_frequency_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved frequency spectrum plot to examples/fourier_frequency_spectrum.png")
    
    fig3 = plot_phase_spectrum(freqs, phase_clean)
    plt.savefig('examples/fourier_phase_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved phase spectrum plot to examples/fourier_phase_spectrum.png")
    
    fig4 = demonstrate_frequency_resolution(fs, duration)
    plt.savefig('examples/fourier_resolution_demo.png', dpi=150, bbox_inches='tight')
    print("Saved resolution demo plot to examples/fourier_resolution_demo.png")
    
    # Show all plots
    plt.show()
    
    print("\n" + "=" * 60)
    print("Fourier analysis complete!")
    print("Key insights:")
    print("- FFT decomposes signals into frequency components")
    print("- Frequency resolution = 1 / duration")
    print("- Noise appears as broadband energy in spectrum")
    print("- Peaks in spectrum correspond to signal frequencies")
    print("=" * 60)


if __name__ == "__main__":
    main()
