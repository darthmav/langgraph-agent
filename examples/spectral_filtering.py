"""
Spectral Filtering Example

Demonstrates low-pass, high-pass, and band-pass filtering in the frequency domain.
Shows how to design and apply filters using FFT.

References:
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing.
- Proakis, J. G., & Manolakis, D. G. (2007). Digital Signal Processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft


def generate_signal_with_components(fs=1000, duration=1.0):
    """
    Generate a signal with low, medium, and high frequency components.
    
    Returns:
    --------
    t : ndarray
        Time array
    signal : ndarray
        Composite signal
    components : dict
        Dictionary with individual frequency components
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Low frequency component (10 Hz)
    low = np.sin(2 * np.pi * 10 * t)
    
    # Medium frequency component (50 Hz)
    medium = 0.5 * np.sin(2 * np.pi * 50 * t)
    
    # High frequency component (150 Hz)
    high = 0.3 * np.sin(2 * np.pi * 150 * t)
    
    # Composite signal
    signal = low + medium + high
    
    components = {
        'low': low,      # 10 Hz
        'medium': medium, # 50 Hz
        'high': high,    # 150 Hz
        'composite': signal
    }
    
    return t, signal, components


def create_lowpass_filter(freqs, cutoff):
    """
    Create ideal low-pass filter.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    cutoff : float
        Cutoff frequency in Hz
    
    Returns:
    --------
    H : ndarray
        Filter frequency response
    """
    H = np.zeros_like(freqs)
    H[np.abs(freqs) <= cutoff] = 1
    return H


def create_highpass_filter(freqs, cutoff):
    """
    Create ideal high-pass filter.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    cutoff : float
        Cutoff frequency in Hz
    
    Returns:
    --------
    H : ndarray
        Filter frequency response
    """
    H = np.ones_like(freqs)
    H[np.abs(freqs) <= cutoff] = 0
    return H


def create_bandpass_filter(freqs, low_cutoff, high_cutoff):
    """
    Create ideal band-pass filter.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    low_cutoff : float
        Lower cutoff frequency in Hz
    high_cutoff : float
        Upper cutoff frequency in Hz
    
    Returns:
    --------
    H : ndarray
        Filter frequency response
    """
    H = np.zeros_like(freqs)
    mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
    H[mask] = 1
    return H


def create_bandstop_filter(freqs, low_cutoff, high_cutoff):
    """
    Create ideal band-stop (notch) filter.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    low_cutoff : float
        Lower cutoff frequency in Hz
    high_cutoff : float
        Upper cutoff frequency in Hz
    
    Returns:
    --------
    H : ndarray
        Filter frequency response
    """
    H = np.ones_like(freqs)
    mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
    H[mask] = 0
    return H


def apply_filter(signal, H):
    """
    Apply filter in frequency domain.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    H : ndarray
        Filter frequency response
    
    Returns:
    --------
    filtered : ndarray
        Filtered signal (real part)
    """
    # Forward FFT
    X = fft.fft(signal)
    
    # Apply filter
    Y = X * H
    
    # Inverse FFT
    filtered = fft.ifft(Y)
    
    return np.real(filtered)


def plot_filter_response(freqs, H, filter_name, ax):
    """
    Plot filter frequency response.
    """
    # Only plot positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    H_pos = H[pos_mask]
    
    ax.plot(freqs_pos, H_pos, linewidth=2, color='darkblue')
    ax.fill_between(freqs_pos, H_pos, alpha=0.3, color='darkblue')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title(filter_name)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)


def plot_filtering_results(t, original, filtered, filter_name, components=None):
    """
    Plot original and filtered signals in time and frequency domains.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Time domain - full signal
    axes[0, 0].plot(t, original, label='Original', alpha=0.7)
    axes[0, 0].plot(t, filtered, label='Filtered', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'{filter_name} - Time Domain')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 0.2)  # Show first 200ms
    
    # Frequency domain
    n = len(original)
    fs = 1 / (t[1] - t[0])
    
    orig_fft = np.abs(fft.fft(original))[:n//2] * 2 / n
    filt_fft = np.abs(fft.fft(filtered))[:n//2] * 2 / n
    freqs = fft.fftfreq(n, d=1/fs)[:n//2]
    
    axes[0, 1].plot(freqs, orig_fft, label='Original', alpha=0.7)
    axes[0, 1].plot(freqs, filt_fft, label='Filtered', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title(f'{filter_name} - Frequency Domain')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 200)
    
    # Time domain - zoomed in
    axes[1, 0].plot(t, original, label='Original', alpha=0.5)
    axes[1, 0].plot(t, filtered, label='Filtered', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title(f'{filter_name} - Zoomed Time Domain')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 0.05)  # Show first 50ms
    
    # Error signal
    error = original - filtered
    axes[1, 1].plot(t, error, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'{filter_name} - Removed Components')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 0.2)
    
    plt.tight_layout()
    return fig


def main():
    """Main demonstration of spectral filtering."""
    print("=" * 60)
    print("Spectral Filtering Demonstration")
    print("=" * 60)
    
    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    duration = 1.0  # Signal duration (seconds)
    
    print(f"\nSignal parameters:")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Nyquist frequency: {fs / 2} Hz")
    
    # Generate signal
    print("\nGenerating signal with components at 10 Hz, 50 Hz, and 150 Hz...")
    t, signal, components = generate_signal_with_components(fs, duration)
    
    # Compute frequency array
    n = len(signal)
    freqs = fft.fftfreq(n, d=1/fs)
    
    # Define filters
    filters = {
        'Low-Pass (30 Hz)': create_lowpass_filter(freqs, 30),
        'High-Pass (40 Hz)': create_highpass_filter(freqs, 40),
        'Band-Pass (40-60 Hz)': create_bandpass_filter(freqs, 40, 60),
        'Band-Stop (45-55 Hz)': create_bandstop_filter(freqs, 45, 55)
    }
    
    # Plot filter responses
    fig_filters, axes_filters = plt.subplots(2, 2, figsize=(14, 8))
    axes_filters = axes_filters.flatten()
    
    for idx, (name, H) in enumerate(filters.items()):
        plot_filter_response(freqs, H, name, axes_filters[idx])
    
    plt.tight_layout()
    plt.savefig('examples/filter_responses.png', dpi=150, bbox_inches='tight')
    print("Saved filter responses to examples/filter_responses.png")
    
    # Apply each filter and visualize results
    print("\nApplying filters...")
    
    for filter_name, H in filters.items():
        print(f"  Applying {filter_name}...")
        filtered = apply_filter(signal, H)
        
        fig = plot_filtering_results(t, signal, filtered, filter_name)
        filename = f"examples/filter_{filter_name.lower().replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"    Saved to {filename}")
        plt.close(fig)
    
    # Demonstrate filter performance metrics
    print("\n" + "=" * 60)
    print("Filter Performance Analysis")
    print("-" * 60)
    
    for filter_name, H in filters.items():
        filtered = apply_filter(signal, H)
        
        # Compute power in original and filtered signals
        original_power = np.sum(signal ** 2)
        filtered_power = np.sum(filtered ** 2)
        removed_power = original_power - filtered_power
        
        print(f"\n{filter_name}:")
        print(f"  Original signal power: {original_power:.3f}")
        print(f"  Filtered signal power: {filtered_power:.3f}")
        print(f"  Power removed: {removed_power:.3f} ({100*removed_power/original_power:.1f}%)")
        
        # Compute correlation with original components
        corr_low = np.corrcoef(filtered, components['low'])[0, 1]
        corr_medium = np.corrcoef(filtered, components['medium'])[0, 1]
        corr_high = np.corrcoef(filtered, components['high'])[0, 1]
        
        print(f"  Correlation with 10 Hz component: {corr_low:.3f}")
        print(f"  Correlation with 50 Hz component: {corr_medium:.3f}")
        print(f"  Correlation with 150 Hz component: {corr_high:.3f}")
    
    # Show filter responses plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("Spectral filtering complete!")
    print("Key insights:")
    print("- Filters modify signal in frequency domain")
    print("- Low-pass keeps low frequencies, removes high")
    print("- High-pass keeps high frequencies, removes low")
    print("- Band-pass isolates specific frequency ranges")
    print("- Ideal filters have sharp transitions (Gibbs phenomenon)")
    print("=" * 60)


if __name__ == "__main__":
    main()
