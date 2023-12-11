import numpy as np
from scipy.signal import chirp, fftconvolve


def pulse_compression(chirp_signal, measured_signal, norm=False):
    """
    Perform pulse compression on a complex signal using a given complex chirp signal with numpy.correlate.
    
    Args:
        signal (numpy.ndarray): The complex signal to be compressed.
        chirp_signal (numpy.ndarray): The complex chirp signal used for compression.
        normalize (bool, optional): Whether to normalize the compressed signal. Default is False.

    Returns:
        numpy.ndarray: The compressed signal (complex).
    """
    # Signals
    c = chirp_signal
    s = measured_signal
    N = len(s)
    
    # Compression
    S = np.fft.fft(s)
    Cconj= np.conj(np.fft.fft(c))
    OUT = S*Cconj
        
    
    # Reformat
    out = np.fft.ifft(OUT)
    out = np.roll(out, int(len(out)/2))
    
    if norm:
        chirp = np.fft.ifft(np.fft.fft(c)*Cconj)
        out = out/np.max(chirp)
    
    return out