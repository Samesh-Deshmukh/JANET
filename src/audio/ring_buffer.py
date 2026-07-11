"""Fixed-size pre-roll ring buffer for audio.

Silero VAD reports "speech started" a few frames after speech actually begins,
so by the time we start collecting, the first phoneme is already gone. This
buffer continuously retains the last PRE_ROLL_MS of audio; when speech starts we
prepend snapshot() to the utterance so the opening word survives.

Pure NumPy, no mic — run `python audio/ring_buffer.py` to exercise the asserts.
"""
import numpy as np

TARGET_RATE = 16000       # samples per second we operate at
PRE_ROLL_MS = 500         # how much audio to keep in front of a detected utterance
PRE_ROLL_SAMPLES = PRE_ROLL_MS * TARGET_RATE // 1000  # = 8000 samples


class RingBuffer:
    """Keeps the most recent `capacity` samples of float32 audio."""

    def __init__(self, capacity=PRE_ROLL_SAMPLES):
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._filled = 0     # how many valid samples we hold (caps at capacity)
        self._end = 0        # index one past the newest sample (write head)

    def push(self, frame):
        """Append a 1-D float32 frame, overwriting the oldest samples when full."""
        frame = np.asarray(frame, dtype=np.float32)
        n = len(frame)
        if n >= self.capacity:
            # frame alone fills/overflows the buffer: keep only its tail
            self._buf[:] = frame[-self.capacity:]
            self._end = 0
            self._filled = self.capacity
            return
        end = self._end
        first = min(n, self.capacity - end)   # part that fits before wrapping
        self._buf[end:end + first] = frame[:first]
        rest = n - first                      # part that wraps to the front
        if rest:
            self._buf[:rest] = frame[first:]
        self._end = (end + n) % self.capacity
        self._filled = min(self._filled + n, self.capacity)

    def snapshot(self):
        """Return retained audio, oldest -> newest, as a float32 array."""
        if self._filled < self.capacity:
            # not yet wrapped: valid data is [0 .. _end)
            return self._buf[:self._end].copy()
        # wrapped: oldest sample sits at _end, read from there around the ring
        return np.concatenate((self._buf[self._end:], self._buf[:self._end]))

    def clear(self):
        self._filled = 0
        self._end = 0


if __name__ == "__main__":
    # 1) Under-capacity: snapshot returns exactly what went in, in order.
    rb = RingBuffer(capacity=10)
    rb.push(np.array([1, 2, 3], dtype=np.float32))
    rb.push(np.array([4, 5], dtype=np.float32))
    assert np.array_equal(rb.snapshot(), [1, 2, 3, 4, 5]), rb.snapshot()

    # 2) Over-capacity across pushes: only the newest `capacity` samples remain.
    rb = RingBuffer(capacity=4)
    rb.push(np.array([1, 2, 3], dtype=np.float32))
    rb.push(np.array([4, 5, 6], dtype=np.float32))   # total 6 pushed, keep last 4
    assert np.array_equal(rb.snapshot(), [3, 4, 5, 6]), rb.snapshot()

    # 3) Single frame larger than capacity keeps its tail.
    rb = RingBuffer(capacity=3)
    rb.push(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert np.array_equal(rb.snapshot(), [3, 4, 5]), rb.snapshot()

    # 4) clear() empties it.
    rb.clear()
    assert rb.snapshot().size == 0

    # 5) Real sizing: 500ms at 16kHz is 8000 samples.
    assert PRE_ROLL_SAMPLES == 8000
    print("✅ ring_buffer self-checks passed")
