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

    # JavaScript function to auto-click the connect button every 60 seconds
    js_code = '''
    function ClickConnect(){
      let btn = document.querySelector("colab-connect-button");
      if (btn != null){
        console.log("✅ Clicked connect button to keep session alive");
        btn.click();
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