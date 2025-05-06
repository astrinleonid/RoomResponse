import numpy as np
import librosa
from tqdm import tqdm


def extract_mfcc_from_array(audio_array, sample_rate=22050, n_mfcc=13, pad_mode='constant'):
    """
    Extract MFCC features from an array of audio signals.

    Args:
        audio_array (numpy.ndarray): 2D NumPy array where each row is an audio signal
        sample_rate (int, optional): Sample rate of the audio signals (Hz)
        n_mfcc (int, optional): Number of MFCC coefficients to extract
        pad_mode (str, optional): Padding mode for shorter signals ('constant', 'reflect', etc.)

    Returns:
        numpy.ndarray: 3D array of MFCC coefficients with shape (n_signals, n_mfcc, n_frames)
    """
    n_signals = audio_array.shape[0]
    mfcc_list = []

    for i in tqdm(range(n_signals)):
        # Get the audio signal and remove NaN padding if any
        signal = audio_array[i]
        mask = ~np.isnan(signal)
        clean_signal = signal[mask] if np.any(mask) else signal

        # Skip empty signals
        if len(clean_signal) == 0:
            print(f"Warning: Signal {i} is empty. Skipping.")
            mfcc_list.append(np.array([]))
            continue

        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=clean_signal,
                sr=sample_rate,
                n_mfcc=n_mfcc
            )
            mfcc_list.append(mfcc)
        except Exception as e:
            print(f"Error extracting MFCC for signal {i}: {e}")
            mfcc_list.append(np.array([]))

    # Find the maximum number of frames across all MFCCs
    max_frames = max([mfcc.shape[1] for mfcc in mfcc_list if mfcc.size > 0], default=0)

    # Create a 3D array to hold all MFCC features
    mfcc_array = np.full((n_signals, n_mfcc, max_frames), np.nan)

    # Fill the array with the extracted MFCC features
    for i, mfcc in enumerate(mfcc_list):
        if mfcc.size > 0:
            mfcc_array[i, :, :mfcc.shape[1]] = mfcc

    return mfcc_array

# Example usage:
# matching_files = extract_files_with_substring('/path/to/folder', 'audio')
# audio_array = combine_files_to_numpy_array(matching_files)
#
# # Extract MFCC features (adjust sample_rate to match your audio files)
# mfcc_array = extract_mfcc_from_array(audio_array, sample_rate=16000, n_mfcc=13)