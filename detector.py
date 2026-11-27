import cv2


class FaceDetector:
    """Detects faces using a Haar Cascade (scaleFactor=1.1, minNeighbors=5, minSize=60x60)."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # إذا فشل تحميل ملف الـ cascade يكون الكائن فارغًا
        if self.cascade.empty():
            raise RuntimeError("Could not load the Haar cascade file.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        """Takes a BGR frame, returns a list of (x, y, w, h)."""
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [tuple(int(v) for v in f) for f in faces]

    def detect_largest(self, frame):
        """يرجع أكبر وجه فقط (x, y, w, h) أو None. نفس القاعدة في التطبيق والتقييم (MD5)."""
        boxes = self.detect(frame)
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])