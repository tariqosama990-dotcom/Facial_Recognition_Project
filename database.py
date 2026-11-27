import os
import pickle
import re
import tempfile

import cv2
import numpy as np

from matcher import make_template


class FaceDatabase:
    """يخزّن لكل شخص قائمة بصمات (samples) ويبني منها قالبًا = المتوسط المطبَّع.
    known_embeddings: اسم -> قالب (هو ما يستخدمه FaceMatcher)."""

    def __init__(self, db_path="face_embeddings.pkl", dataset_dir="dataset", embedding_size=512):
        self.db_path = db_path
        self.dataset_dir = dataset_dir
        self.embedding_size = embedding_size
        self.samples = {}            # اسم -> [بصمات]
        self.known_embeddings = {}   # اسم -> قالب

        os.makedirs(self.dataset_dir, exist_ok=True)
        self.load()

    # ---------- التحميل ----------
    def load(self):
        self.samples = {}
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    loaded = pickle.load(f)
            except Exception as e:  # ملف تالف
                print(f"Could not read {self.db_path} ({e}). Starting with an empty database.")
                loaded = {}

            for name, value in loaded.items():
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1:            # الصيغة القديمة: بصمة واحدة لكل شخص
                    arr = arr[np.newaxis, :]
                if arr.ndim == 2 and arr.shape[1] == self.embedding_size:
                    self.samples[name] = [row for row in arr]
                else:
                    print(f"Ignored [{name}]: embedding size is incompatible "
                          f"with the current model ({self.embedding_size}).")
            print(f"Loaded {len(self.samples)} people from memory.")
        else:
            print("The database is currently empty.")
        self._rebuild_templates()

    def _rebuild_templates(self):
        self.known_embeddings = {n: make_template(s) for n, s in self.samples.items() if s}

    # ---------- الحفظ ----------
    @staticmethod
    def _safe_name(name):
        """يمنع الرموز الخطرة في اسم الملف (مثل ../ أو :)"""
        return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_") or "person"

    def _next_image_path(self, name):
        base = self._safe_name(name)
        i = 1
        while os.path.exists(os.path.join(self.dataset_dir, f"{base}_{i}.jpg")):
            i += 1
        return os.path.join(self.dataset_dir, f"{base}_{i}.jpg")

    def _persist(self):
        """كتابة ذرّية: ملف مؤقت ثم استبدال، حتى لا يتلف الملف عند انقطاع الكتابة."""
        directory = os.path.dirname(os.path.abspath(self.db_path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.samples, f)
            os.replace(tmp, self.db_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def save_person(self, name, face_rgb, embedding):
        """يحفظ صورة الوجه (RGB 160x160) ويضيف البصمة إلى عينات الشخص (لا يستبدلها).
        يرجع عدد عينات الشخص بعد الإضافة."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.embedding_size,):
            raise ValueError(f"Embedding must have shape ({self.embedding_size},).")

        image_path = self._next_image_path(name)
        image_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        previous_samples = self.samples.get(name)
        previous_templates = self.known_embeddings
        new_samples = [*(previous_samples or []), embedding]

        try:
            if not cv2.imwrite(image_path, image_bgr):
                raise OSError(f"Could not write face image: {image_path}")

            self.samples[name] = new_samples
            self._rebuild_templates()
            self._persist()
        except Exception:
            if previous_samples is None:
                self.samples.pop(name, None)
            else:
                self.samples[name] = previous_samples
            self.known_embeddings = previous_templates
            try:
                os.remove(image_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                print(f"Could not remove {image_path}: {cleanup_error}")
            raise

        return len(new_samples)

    def delete_person(self, name, delete_images=True):
        """يحذف الشخص وبصماته من القاعدة (وصور وجهه من dataset/ اختياريًا).
        يرجع عدد الصور المحذوفة، أو None إذا لم يكن الاسم موجودًا."""
        if name not in self.samples:
            return None

        del self.samples[name]
        self._rebuild_templates()
        self._persist()          # نحفظ القاعدة أولاً، ثم نحذف الصور

        removed = 0
        if delete_images:
            # نطابق الاسم بدقة: Ali_1.jpg لا يطابق Ali_1_1.jpg (شخص اسمه "Ali_1")
            pattern = re.compile(rf"^{re.escape(self._safe_name(name))}_\d+\.jpg$")
            for fn in os.listdir(self.dataset_dir):
                if pattern.match(fn):
                    try:
                        os.remove(os.path.join(self.dataset_dir, fn))
                        removed += 1
                    except OSError as e:
                        print(f"Could not delete {fn}: {e}")
        return removed
