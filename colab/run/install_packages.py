# Cell 1: Install packages - identical on Colab and Kaggle, no
# platform-specific packages here (unsloth needs CUDA, which both
# platforms' GPU runtimes provide).

!pip install unsloth fastapi uvicorn pyngrok nest-asyncio

# ===== ADD THIS AT THE END OF CELL 1 =====
import time
print("\n⏳ Verifying package installation...")
time.sleep(5)  # Wait a bit for pip to complete

# Try importing to verify
try:
    import unsloth
    import fastapi
    import uvicorn
    import pyngrok
    import nest_asyncio
    print("✅ All packages verified and ready!")
except ImportError as e:
    print(f"⚠️ Package import failed: {e}")
    print("Waiting additional 5 seconds...")
    time.sleep(5)

print("✅ Cell 1 complete. Proceed to Cell 2.")
# =========================================