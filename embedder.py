import cv2
import numpy as np
from keras_facenet import FaceNet


class FaceEmbedder:
    def __init__(self, input_size=(160, 160)):
        # keras-facenet يطبّق (pixel-127.5)/127.5 داخليًا للنموذج الافتراضي
        self.model = FaceNet()
        self.input_size = input_size

    @staticmethod
    def crop(frame_bgr, box):
        """يقصّ الوجه مع حصر الإحداثيات داخل الصورة. يرجع None إذا كان القص فارغًا."""
        if frame_bgr is None or box is None:
            return None
        x, y, w, h = box
        H, W = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame_bgr[y0:y1, x0:x1]

    def preprocess(self, face_bgr):
        """BGR -> RGB ثم تغيير الحجم إلى 160x160"""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, self.input_size)

    def embed(self, face_bgr):
        """ترجع متجه البصمة (512,) لوجه مقصوص"""
        img = self.preprocess(face_bgr)
        return self.model.embeddings(np.expand_dims(img, axis=0))[0]

    def embed_from_frame(self, frame_bgr, box):
        face = self.crop(frame_bgr, box)
        if face is None or face.size == 0:
            return None
        return self.embed(face)