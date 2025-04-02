from app import create_app
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
