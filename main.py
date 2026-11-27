from detector import FaceDetector
from embedder import FaceEmbedder
from matcher import FaceMatcher
from database import FaceDatabase
from gui import FaceApp

def main():
    print("Loading the FaceNet model...")
    embedder = FaceEmbedder()
    print("The model has been successfully loaded.")

    try:
        app = FaceApp(
            detector=FaceDetector(),
            embedder=embedder,
            matcher=FaceMatcher(),
            database=FaceDatabase(),
        )
    except RuntimeError as e:   # مثل فشل فتح الكاميرا
        raise SystemExit(f"Startup error: {e}")
    app.run()


if __name__ == "__main__":
    main()