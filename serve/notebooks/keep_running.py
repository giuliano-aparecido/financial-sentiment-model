# Cell 4: Keep the session alive so ngrok/FastAPI keep serving requests.
#
# The DOM-click trick below is Colab-only - it clicks Colab's own
# "colab-connect-button", which doesn't exist on Kaggle, so it would
# silently do nothing there. Kaggle's idle/session model isn't the same
# shape to begin with (1hr idle timeout on the interactive session, 9hr
# hard cap either way - https://www.kaggle.com/product-feedback/52499),
# and its actual fix isn't a script: use "Save Version -> Save & Run All
# (Commit)" instead of running interactively, so the notebook runs as a
# background batch session that isn't subject to the idle timeout at all.

try:
    import google.colab
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

import threading
import time

if not IS_COLAB:
    print("Not running on Colab - the auto-reconnect trick below doesn't apply here.")
    print("On Kaggle: use 'Save Version -> Save & Run All (Commit)' instead of running")
    print("this interactively - that runs as a background batch session (up to 9h)")
    print("that isn't subject to the interactive idle timeout this cell works around.")
else:
    from IPython.display import display, Javascript

    # JavaScript function to auto-click the connect button every 60 seconds.
    #
    # Confirmed (multiple independent reports, not verified against a live
    # session here - see the calling context for how this was diagnosed):
    # <colab-connect-button> is a custom element with a SHADOW DOM. The old
    # version below just did document.querySelector("colab-connect-button")
    # and called .click() on THAT element - it exists (btn != null passes,
    # no error, the console message even prints), but the real interactive
    # button lives inside btn.shadowRoot, so the click never reached
    # anything real. That's a plausible root cause for "keep_running.py
    # runs with no errors, but the session still dies around the 90-minute
    # idle mark" - the click was always a no-op.
    #
    # Reaches into the shadow root first; falls back to the old top-level
    # click if that structure isn't there (a future Colab UI change, or
    # this diagnosis being wrong for some other reason) - so this degrades
    # to the previous (silently ineffective) behavior rather than throwing.
    # Watch the browser console: "(shadow DOM)" confirms the fix path is
    # actually firing; "(fallback)" or the "not found" warning means
    # Colab's DOM has changed again and the selector needs updating by
    # hand (right-click the connect button -> Inspect -> find the current
    # structure) - there's no way to verify this without a live session.
    js_code = '''
    function ClickConnect(){
      try {
        let outer = document.querySelector("#top-toolbar > colab-connect-button")
                 || document.querySelector("colab-connect-button");
        let shadowBtn = outer && outer.shadowRoot && outer.shadowRoot.querySelector("#connect");
        if (shadowBtn) {
          shadowBtn.click();
          console.log("✅ Clicked connect button (shadow DOM) to keep session alive");
        } else if (outer) {
          outer.click();
          console.log("✅ Clicked connect button (fallback, no shadow DOM found) to keep session alive");
        } else {
          console.log("⚠️ Connect button not found - Colab's UI may have changed again, inspect the page to find the current selector");
        }
      } catch (e) {
        console.log("⚠️ ClickConnect error: " + e);
      }
    }
    // Click every 60 seconds (60000 milliseconds)
    setInterval(ClickConnect, 60000);
    console.log("🔄 Auto-reconnect enabled - session will stay alive");
    '''

    # Display the JavaScript in the notebook
    display(Javascript(js_code))

    print("=" * 60)
    print("✅ SESSION KEEPER ACTIVATED")
    print("=" * 60)
    print("The notebook will auto-click every 60 seconds to prevent")
    print("the 90-minute idle timeout.")
    print("\n⚠️  IMPORTANT LIMITS:")
    print("   • Idle timeout: 90 minutes (THIS IS PREVENTED)")
    print("   • Maximum session length: 12 hours (CANNOT BE PREVENTED)")
    print("   • After 12 hours, you MUST restart Colab")
    print("\n📍 When restarting:")
    print("   1. Run all cells again")
    print("   2. Cell 3 will auto-send new ngrok URL to Render")
    print("   3. Cell 4 will keep the new session alive")
    print("=" * 60)

# Optional: Python-side monitor (prints status every 60 seconds) - runs on
# either platform, though on Kaggle it's just a heartbeat print, not an
# actual idle-timeout workaround (see note above).
def python_monitor():
    """Print status to show the session is active"""
    count = 0
    while True:
        time.sleep(60)
        count += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Session alive - {count} min(s) elapsed")

# Start monitor in background
monitor_thread = threading.Thread(target=python_monitor, daemon=True)
monitor_thread.start()