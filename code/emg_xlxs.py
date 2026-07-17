import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

class EMGProcessor:
    def __init__(self, csv_path: str, fs: int = 2000):
        self.csv_path = csv_path
        self.fs = fs
        self.time = None
        self.rms_df = None
        self.rms_emg_normalized = None

    def load_data(self):
        """
        Loads CSV EMG data using the exact column names provided earlier:
            - BICEPS SH RT (uV)
            - BICEPS BR. RT (uV)
            - LAT. TRICEPS RT (uV)
            - MED. TRICEPS RT (uV)
        """

        df = pd.read_csv(self.csv_path)
        print(df.columns)

        # Extract columns (IMPORTANT: These names must exist in CSV)
        muscle1 = df["BICEPS SH RT (uV)"].values
        muscle2 = df["BICEPS BR. RT (uV)"].values
        muscle3 = df["LAT. TRICEPS RT (uV)"].values
        muscle4 = df["MED. TRICEPS RT (uV)"].values

        min_len = max(len(muscle1), len(muscle2), len(muscle3), len(muscle4))

        # Time calculation based on fs
        self.time = np.arange(min_len) / self.fs

        self.emg_signals = pd.DataFrame({
            'Time': self.time,
            'Muscle1': muscle1[:min_len],
            'Muscle2': muscle2[:min_len],
            'Muscle3': muscle3[:min_len],
            'Muscle4': muscle4[:min_len]
        })

        print("✅ CSV Loaded Successfully")

    def compute_psd_bands(self):
        emg_data = self.emg_signals.iloc[:, 1:].values
        self.min_band = []
        self.max_band = []

        for i in range(4):
            freqs, psd = welch(emg_data[:, i], fs=self.fs, nperseg=2048)

            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]
            threshold = 0.1 * psd[peak_idx]
            active_band = freqs[psd >= threshold]

            if len(active_band) > 0:
                self.min_band.append(active_band[0])
                self.max_band.append(active_band[-1])

            idx_50 = np.argmin(np.abs(freqs - 50))
            power_50hz = psd[idx_50]
            print(f"Muscle {i+1}: Peak @ {peak_freq:.2f} Hz, 50Hz Power = {power_50hz:.2e}")

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    def notch_filter(self, data, freq, Q=30):
        nyq = 0.5 * self.fs
        w0 = freq / nyq
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data)

    def apply_filters(self):
        emg_data = self.emg_signals.iloc[:, 1:].values
        lowcut = np.mean(self.min_band)
        highcut = np.mean(self.max_band)

        print(f"✔ Bandpass Range = {lowcut:.2f} – {highcut:.2f} Hz")

        filtered_emg = []
        for i in range(emg_data.shape[1]):
            raw = emg_data[:, i]
            bandpassed = self.bandpass_filter(raw, lowcut, highcut)
            cleaned = self.notch_filter(bandpassed, 50)   # Remove 50 Hz noise
            filtered_emg.append(cleaned)

        self.filtered_emg = np.array(filtered_emg).T

    def calculate_rms(self, signal, window_size):
        return np.sqrt(np.convolve(signal**2, np.ones(window_size) / window_size, mode='same'))

    def compute_rms(self):
        window_size = int(0.05 * self.fs)  # 50 ms window
        rms_emg = []

        for i in range(self.filtered_emg.shape[1]):
            rectified = np.abs(self.filtered_emg[:, i])
            rms = self.calculate_rms(rectified, window_size)
            rms_emg.append(rms)

        self.rms_emg = np.array(rms_emg).T

    def normalize_rms(self):
        rms_emg_normalized = np.zeros_like(self.rms_emg)
        for i in range(self.rms_emg.shape[1]):
            min_val = np.min(self.rms_emg[:, i])
            max_val = np.max(self.rms_emg[:, i])
            rms_emg_normalized[:, i] = (self.rms_emg[:, i] - min_val) / (max_val - min_val)

        self.rms_emg_normalized = rms_emg_normalized

    def save_csv(self, filename='output.csv'):
        self.rms_df = pd.DataFrame(self.rms_emg_normalized, columns=[f'Muscle{i+1}' for i in range(4)])
        self.rms_df.insert(0, 'Time', self.time)
        self.rms_df.to_csv(filename, index=False)
        print(f"💾 Saved normalized RMS to {filename}")

    def run_all(self):
        self.load_data()
        self.compute_psd_bands()
        self.apply_filters()
        self.compute_rms()
        self.normalize_rms()
        return self.rms_emg_normalized


if __name__ == "__main__":

    csv_path = r"D:\books\IIT D\3rd\Mtp\Loading_unloading_data\Loading_unloading_data\EMG data\Ravinder_7.5KG_up.csv"

    processor = EMGProcessor(csv_path)
    final_output = processor.run_all()

    print("\nFinal RMS Normalized Output Shape:", final_output.shape)
