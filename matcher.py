import numpy as np
from scipy.spatial.distance import cosine


def make_template(embeddings):
    """يبني قالب الشخص = المتوسط المطبَّع (L2) لبصماته، كما في المنهجية (MD7).
    يُستخدم نفسه في التطبيق وفي evaluate.py حتى لا يختلف السلوك بينهما."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:                      # بصمة واحدة فقط
        arr = arr[np.newaxis, :]
    if arr.shape[0] == 0:
        raise ValueError("Cannot build a template from zero embeddings.")
    template = arr.mean(axis=0)
    norm = np.linalg.norm(template)
    if norm == 0:
        raise ValueError("Template has zero norm.")
    return template / norm


DEFAULT_THRESHOLD = 0.40


class FaceMatcher:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold

    def recognize(self, live_vector, db):
        """db: dict اسم -> قالب. ترجع (identity, min_dist)"""
        # لا يوجد وجه أو لا يوجد أحد مسجّل بعد -> Unknown
        if live_vector is None or not db:
            return "Unknown", 1.0

        min_dist = float("inf")
        identity = "Unknown"

        # نقارن بكل القوالب ونأخذ الأقرب (أصغر مسافة cosine)
        for name, saved_vector in db.items():
            dist = cosine(live_vector, saved_vector)
            if dist < min_dist:
                min_dist = dist
                identity = name

        # إذا تجاوزت المسافة العتبة نعتبره غريبًا
        if min_dist > self.threshold:
            identity = "Unknown"

        return identity, min_dist