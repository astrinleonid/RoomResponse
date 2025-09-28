import threading
import queue
import time
import numpy as np

_SDL_AVAILABLE = False
try:
    import sdl_audio_core
    _SDL_AVAILABLE = True
except ImportError:
    sdl = None
    _SDL_AVAILABLE = False


class AudioRecorder:
    """Reusable audio recording component"""

    def __init__(self, sample_rate: int = 48000, input_device: int = -1, enable_logging: bool = False):
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.enable_logging = enable_logging
        self.engine = None

    def __enter__(self):
        """Context manager entry - initialize engine"""


        self.engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = self.sample_rate
        config.enable_logging = self.enable_logging

        if not self.engine.initialize(config):
            raise RuntimeError("Failed to initialize audio engine")

        if self.input_device >= 0:
            self.engine.set_input_device(self.input_device)

        if not self.engine.start():
            raise RuntimeError("Failed to start audio engine")

        if not self.engine.start_recording():
            raise RuntimeError("Failed to start recording")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        if self.engine:
            self.engine.stop_recording()
            self.engine.stop()
            self.engine.shutdown()

    def get_audio_chunk(self, min_samples: int) -> np.ndarray:
        """Get audio chunk with at least min_samples"""
        if not self.engine:
            return np.array([])

        recorded_data = self.engine.get_recorded_data()
        if len(recorded_data) >= min_samples:
            latest_chunk = recorded_data[-min_samples:]
            self.engine.clear_recording_buffer()
            return np.array(latest_chunk, dtype=np.float32)
        return np.array([])


class AudioProcessor:
    """Reusable audio processing component"""

    @staticmethod
    def calculate_rms(audio_chunk: np.ndarray) -> float:
        """Calculate RMS of audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk ** 2))

    @staticmethod
    def rms_to_db(rms: float) -> float:
        """Convert RMS to dB scale"""
        return 20 * np.log10(max(rms, 1e-8))

    @staticmethod
    def create_level_meter(rms_db: float, width: int = 50, range_db: int = 60) -> str:
        """Create visual level meter"""
        level_bars = int(min(width, max(0, (rms_db + range_db) * width / range_db)))
        return '█' * level_bars + '░' * (width - level_bars)


class AudioRecordingWorker:
    """Worker thread for continuous audio recording"""

    def __init__(self, recorder: AudioRecorder, output_queue: queue.Queue,
                 chunk_duration: float = 0.1):
        self.recorder = recorder
        self.output_queue = output_queue
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * recorder.sample_rate)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Start the recording worker thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 2.0):
        """Stop the recording worker thread"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=timeout)

    def _run(self):
        """Main recording loop"""
        try:
            while not self.stop_event.is_set():
                time.sleep(self.chunk_duration)

                audio_chunk = self.recorder.get_audio_chunk(self.chunk_samples)
                if len(audio_chunk) > 0:
                    self.output_queue.put(audio_chunk)

        except Exception as e:
            print(f"Recording worker error: {e}")


class AudioProcessingWorker:
    """Worker thread for processing audio chunks"""

    def __init__(self, input_queue: queue.Queue, processor_func=None):
        self.input_queue = input_queue
        self.processor_func = processor_func or self._default_processor
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Start the processing worker thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 2.0):
        """Stop the processing worker thread"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=timeout)

    def _run(self):
        """Main processing loop"""
        try:
            while not self.stop_event.is_set() or not self.input_queue.empty():
                try:
                    audio_chunk = self.input_queue.get(timeout=1.0)
                    self.processor_func(audio_chunk)
                    self.input_queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Processing worker error: {e}")

    def _default_processor(self, audio_chunk: np.ndarray):
        """Default processor - just calculate and display RMS"""
        rms = AudioProcessor.calculate_rms(audio_chunk)
        rms_db = AudioProcessor.rms_to_db(rms)
        meter = AudioProcessor.create_level_meter(rms_db)

        timestamp = time.strftime("%H:%M:%S")
        print(f"\r{timestamp} RMS: {rms:.6f} ({rms_db:+5.1f}dB) |{meter}|", end='', flush=True)




# Add these methods to RoomResponseRecorder class:
