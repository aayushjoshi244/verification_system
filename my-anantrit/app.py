# app.py
import time
from head.mouth import speak, speak_blocking, shutdown
from head.ear import listen

def main():
    # Sanity check: proves TTS path works before the loop
    try:
        speak_blocking("Audio check. I am ready.")
    except Exception as e:
        print(f"🔇 TTS failed before loop: {e}")

    empty_streak = 0
    while True:
        try:
            said = listen("Say Something:")
        except KeyboardInterrupt:
            # Ctrl+C during listen → break loop and shutdown
            print("\n⛔ Interrupted. Exiting loop.")
            break

        if not said:
            empty_streak += 1
            # Pause a moment so we don't spin the CPU
            time.sleep(0.25)
            # Optional: auto-exit after repeated silence
            if empty_streak >= 8:
                speak_blocking("No input detected. Goodbye.")
                break
            continue

        empty_streak = 0

        if said in {"quit", "exit", "stop"}:
            speak_blocking("Goodbye!")
            break

        speak_blocking(f"You said: {said}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🧹 Shutting down...")
    finally:
        shutdown()
