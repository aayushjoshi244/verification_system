# head/ear.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
import speech_recognition as sr

@dataclass
class ASRSettings:
    language: str = "en-IN"
    timeout: float = 4.0
    phrase_time_limit: float = 7.0
    initial_calibration_sec: float = 0.6
    recalibrate_every_sec: float = 300.0  # 5 minutes
    pause_threshold: float = 0.7
    dynamic_energy_threshold: bool = True
    energy_threshold: int = 300
    retries: int = 2

class Ear:
    def __init__(self, settings: Optional[ASRSettings] = None):
        self.settings = settings or ASRSettings()
        self._r = sr.Recognizer()
        self._r.dynamic_energy_threshold = self.settings.dynamic_energy_threshold
        self._r.energy_threshold = self.settings.energy_threshold
        self._r.pause_threshold = self.settings.pause_threshold
        self._r.operation_timeout = None
        self._last_calibration_ts = 0.0
        # Do a quick one-off calibration to avoid clipping first syllables
        try:
            with sr.Microphone() as source:
                print("🎙️  Calibrating mic (quick)...", end="\r", flush=True)
                self._r.adjust_for_ambient_noise(source, duration=self.settings.initial_calibration_sec)
                self._last_calibration_ts = time.time()
                print(" " * 40, end="\r")  # clear line
        except Exception as e:
            # If calibration fails (no mic, busy device), keep defaults
            print(f"⚠️ Mic calibration skipped: {e}")

    def _recalibrate_if_needed(self):
        now = time.time()
        if (now - self._last_calibration_ts) < self.settings.recalibrate_every_sec:
            return
        try:
            with sr.Microphone() as source:
                print("🎙️  Recalibrating mic...", end="\r", flush=True)
                self._r.adjust_for_ambient_noise(source, duration=0.4)  # very short
                self._last_calibration_ts = time.time()
                print(" " * 40, end="\r")
        except Exception:
            # Non-fatal: just skip
            pass

    def listen_once(self, prompt: Optional[str] = None) -> str:
        if prompt:
            print(prompt, flush=True)

        attempts = 0
        while attempts <= self.settings.retries:
            attempts += 1
            try:
                self._recalibrate_if_needed()
                with sr.Microphone() as source:
                    print("🎧 Listening... speak now      ", end="\r", flush=True)
                    audio = self._r.listen(
                        source,
                        timeout=self.settings.timeout,
                        phrase_time_limit=self.settings.phrase_time_limit,
                    )
                    print("🧠 Recognizing...               ", end="\r", flush=True)

                text = self._r.recognize_google(audio, language=self.settings.language).strip().lower()
                print(f"✅ Heard: {text:<40}", flush=True)
                return text

            except KeyboardInterrupt:
                print("\n⛔ Interrupted by user.")
                return ""
            except sr.WaitTimeoutError:
                msg = "⏳ No speech detected."
                print(msg + (" Retrying..." if attempts <= self.settings.retries else ""), flush=True)
                continue
            except sr.UnknownValueError:
                msg = "🤷 Could not understand."
                print(msg + (" Retrying..." if attempts <= self.settings.retries else ""), flush=True)
                continue
            except sr.RequestError as e:
                print(f"🚨 ASR service unavailable: {e}", flush=True)
                return ""
            except Exception as e:
                print(f"⚠️ ASR error: {e}", flush=True)
                return ""
        return ""

# Convenience
_default_ear: Optional[Ear] = None
def get_ear() -> Ear:
    global _default_ear
    if _default_ear is None:
        _default_ear = Ear()
    return _default_ear

def listen(prompt: Optional[str] = None) -> str:
    return get_ear().listen_once(prompt)
