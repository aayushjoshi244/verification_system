# head/mouth.py
# Non-blocking TTS with Edge-TTS ( primary ) + optional gTTS fallback.
# Playback via a single background worker using pygame.mixer.
# Safe for use inside agent event loops (no asyncio.run() in main thread).

from __future__ import annotations
import asyncio
import threading
import queue
import tempfile
import uuid
import os
from dataclasses import dataclass
from typing import Optional

# Edge-TTS (online, fast, high quality)
import edge_tts

# Playback
import pygame

# Optional fallback (offline-ish but still needs Google; leave disabled by default)
try:
    from gtts import gTTS  # type: ignore
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False


@dataclass
class TTSSettings:
    voice: str = "en-AU-WilliamNeural"   # set your preferred voice
    rate: str = "0%"                     # "-25%" to "+50%" etc.
    volume: str = "+0%"                  # "-50%".."+50%"
    use_gtts_fallback: bool = False      # fallback if edge-tts fails


class AudioPlayer:
    """Background audio worker that plays queued mp3 files and deletes them after."""
    def __init__(self, buffer_size: int = 1024):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._buffer_size = buffer_size
        self._worker.start()
        # wait for mixer to init
        self._ready.wait(timeout=5.0)

    def _run(self):
        pygame.mixer.pre_init()  # let pygame choose sensible defaults
        pygame.init()
        pygame.mixer.init()
        self._ready.set()

        try:
            while not self._stop.is_set():
                try:
                    path = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue

                try:
                    sound = pygame.mixer.Sound(path)
                    channel = sound.play()
                    # Poll until finished or stop requested
                    while channel.get_busy() and not self._stop.is_set():
                        pygame.time.wait(15)
                finally:
                    # Best-effort cleanup
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        finally:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
                pygame.quit()
            except Exception:
                pass

    def enqueue(self, path: str):
        self._q.put(path)

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)


class Mouth:
    """High-level TTS facade: speak (non-blocking), speak_blocking, stop/close."""
    def __init__(self, settings: Optional[TTSSettings] = None):
        self.settings = settings or TTSSettings()
        self._player = AudioPlayer()
        # Dedicated Edge-TTS event loop in its own thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _temp_mp3(self) -> str:
        return os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")

    async def _edge_tts_to_file(self, text: str, out_path: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.settings.voice,
            rate=self.settings.rate,
            volume=self.settings.volume,
        )
        await communicate.save(out_path)

    def _gtts_to_file(self, text: str, out_path: str):
        if not _HAS_GTTS:
            raise RuntimeError("gTTS not installed; cannot fallback.")
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(out_path)

    def _play_mp3_blocking(self, path: str):
        """Play an MP3 and block until it starts and finishes."""
        # Ensure pygame is initialized (player thread has already done this,
        # but this makes it resilient if called before AudioPlayer enqueue ever ran)
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.pre_init()
                pygame.mixer.init()
            except Exception as e:
                print(f"🔇 Mixer init failed in blocking playback: {e}")
                return

        try:
            sound = pygame.mixer.Sound(path)
            ch = sound.play()
            # Wait until playback actually starts
            waited = 0
            while (ch is None or not ch.get_busy()) and waited < 5000:
                pygame.time.wait(10)
                waited += 10
                ch = ch or sound.play()

            # Now wait for completion
            while ch and ch.get_busy():
                pygame.time.wait(20)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def _edge_tts_to_file_sync(self, text: str, out_path: str):
        """Run edge-tts save synchronously in this thread."""
        # edge-tts is async API; we run its coroutine here
        async def _go():
            await self._edge_tts_to_file(text, out_path)
        asyncio.run(_go())
    
    def speak(self, text: str):
        """Non-blocking speak with robust scheduling even during teardown."""
        if not text or not text.strip():
            return
        out_path = self._temp_mp3()

        async def work():
            try:
                await self._edge_tts_to_file(text, out_path)
            except Exception:
                if self.settings.use_gtts_fallback:
                    try:
                        self._gtts_to_file(text, out_path)
                    except Exception:
                        # Cleanup if created
                        if os.path.exists(out_path):
                            try: os.remove(out_path)
                            except Exception: pass
                        raise
                else:
                    if os.path.exists(out_path):
                        try: os.remove(out_path)
                        except Exception: pass
                    raise
            self._player.enqueue(out_path)

        try:
            # Ensure loop is alive before scheduling
            if not self._loop.is_running():
                # Rare: loop not started or already stopped -> do synchronous generation
                asyncio.run(self._edge_tts_to_file(text, out_path))
                self._player.enqueue(out_path)
                return
            fut = asyncio.run_coroutine_threadsafe(work(), self._loop)
            # Optional: attach a done callback that swallows exceptions (avoids "never retrieved")
            def _swallow(_f): 
                try: _f.result()
                except Exception as _e: print(f"🔇 TTS error: {_e}")
            fut.add_done_callback(_swallow)
        except RuntimeError:
            # Event loop closed during shutdown -> synchronous fallback
            try:
                asyncio.run(self._edge_tts_to_file(text, out_path))
                self._player.enqueue(out_path)
            except Exception as e:
                print(f"🔇 TTS finalization error: {e}")

    def speak_blocking(self, text: str):
        """Guaranteed blocking speak: sync generate + sync playback (no queue)."""
        if not text or not text.strip():
            return
        out_path = self._temp_mp3()
        try:
            # Try Edge-TTS sync generation first
            self._edge_tts_to_file_sync(text, out_path)
        except Exception as e:
            print(f"🔇 Edge-TTS blocking generation failed: {e}")
            # Last resort: try to use the async path then wait for busy
            # (still better than giving up silently)
            try:
                self.speak(text)
                # Wait up to ~7.5s for playback to start, then wait until it’s done
                waited = 0
                # Phase 1: wait until mixer starts playing something
                while not pygame.mixer.get_busy() and waited < 7500:
                    pygame.time.wait(25)
                    waited += 25
                # Phase 2: wait until playback finishes
                while pygame.mixer.get_busy():
                    pygame.time.wait(25)
                return
            except Exception as e2:
                print(f"🔇 Fallback blocking path failed: {e2}")
                return

        # If generation succeeded, play and truly block
        self._play_mp3_blocking(out_path)

    def close(self):
        """Clean shutdown—call at program exit."""
        try:
            self._player.close()
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=2.0)
            except Exception:
                pass


# Convenience, module-level singleton if you like:
_default_mouth: Optional[Mouth] = None

def get_mouth() -> Mouth:
    global _default_mouth
    if _default_mouth is None:
        _default_mouth = Mouth()
    return _default_mouth

def speak(text: str):
    """Drop-in API compatible with your previous code (non-blocking)."""
    get_mouth().speak(text)

def speak_blocking(text: str):
    get_mouth().speak_blocking(text)

def shutdown():
    m = get_mouth()
    m.close()
