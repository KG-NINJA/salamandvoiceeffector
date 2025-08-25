import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal, linalg


def vlmish_ffmpeg(input_path: str = "input.wav", output_path: str = "vlmish1.wav") -> None:
    """Apply a crude VLM-5030 style effect using FFmpeg filters."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af",
        "highpass=f=300, lowpass=f=3300, aresample=8000, asetrate=8000",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def vlmish_scipy(input_path: str = "input.wav", output_path: str = "vlmish2.wav") -> None:
    """Bandpass and quantize the signal using SciPy/NumPy."""
    fs, data = wavfile.read(input_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    b, a = signal.butter(4, [300 / (fs / 2), 3300 / (fs / 2)], btype="band")
    filtered = signal.lfilter(b, a, data)
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered /= max_val
    quantized = np.round(filtered * 127) / 127
    out = np.int16(quantized / np.max(np.abs(quantized)) * 32767)
    wavfile.write(output_path, fs, out)


def _lpc_coeffs(signal_data: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients using autocorrelation and Levinson-Durbin."""
    autocorr = np.correlate(signal_data, signal_data, mode="full")[len(signal_data) - 1 :]
    R = linalg.toeplitz(autocorr[:order])
    r = autocorr[1 : order + 1]
    a = np.linalg.solve(R, r)
    return np.concatenate(([1.0], -a))


def vlmish_lpc(
    input_path: str = "input.wav", output_path: str = "vlmish3.wav", order: int = 16
) -> None:
    """Very simple LPC analysis/resynthesis with residual quantization."""
    fs, data = wavfile.read(input_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    a = _lpc_coeffs(data, order)
    residual = signal.lfilter(a, [1.0], data)
    residual = np.round(residual * 127) / 127
    reconstructed = signal.lfilter([1.0], a, residual)
    out = np.int16(reconstructed / np.max(np.abs(reconstructed)) * 32767)
    wavfile.write(output_path, fs, out)


def main() -> None:
    input_path = "input.wav"
    if os.path.exists(input_path):
        vlmish_ffmpeg(input_path, "vlmish1.wav")
        vlmish_scipy(input_path, "vlmish2.wav")
        vlmish_lpc(input_path, "vlmish3.wav")
    else:
        print(f"Input file {input_path} not found. Place a WAV file named input.wav in the repository root.")


if __name__ == "__main__":
    main()
