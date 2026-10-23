import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from keras_facenet import FaceNet

# ==========================================
# 1. تحميل نموذج FaceNet
# ==========================================
print("Loading FaceNet model...")
embedder = FaceNet()

# ==========================================
# 2. تحميل LFW بدون قص وبحجم كامل
# ==========================================
print("Loading LFW dataset...")
lfw = fetch_lfw_people(
    min_faces_per_person=20,
    color=True,
    resize=1.0,
    slice_=(slice(0, 250), slice(0, 250)),   # الصورة كاملة بدون قص ضيق
    funneled=True
)
images = lfw.images
labels = lfw.target
names = lfw.target_names
print(f"Images: {len(images)} | People: {len(names)}")

# ==========================================
# 3. تجهيز الصور بالشكل الصحيح + استخراج البصمات
# ==========================================
def prepare(img):
    # تحويل إلى uint8 بقيم 0-255 (إذا كانت القيم بين 0 و1 نضربها)
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.astype("uint8")
    # قص مركزي للوجه ثم تغيير الحجم
    h, w = img.shape[:2]
    c = int(min(h, w) * 0.65)
    y0, x0 = (h - c) // 2, (w - c) // 2
    face = img[y0:y0 + c, x0:x0 + c]
    return cv2.resize(face, (160, 160))

print("Extracting embeddings (batch mode)...")
faces = np.array([prepare(im) for im in images])

BATCH = 64
embeddings = []
for i in range(0, len(faces), BATCH):
    batch = faces[i:i + BATCH]
    # embeddings() على دفعة كاملة؛ نستخدم النموذج مباشرة لتجنب اكتشاف الوجه
    emb = embedder.embeddings(batch)
    embeddings.append(emb)
    print(f"Processed {min(i + BATCH, len(faces))} / {len(faces)}")

X = np.vstack(embeddings)

# ==========================================
# 4. تقسيم متوازن (stratify) للتدريب والاختبار
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# ==========================================
# 5. التسوية L2
# ==========================================
norm = Normalizer(norm="l2")
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)

# ==========================================
# 6. التدريب مع البحث عن أفضل المعاملات
# ==========================================
print("\nTraining with GridSearch...")
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale"],
}
grid = GridSearchCV(
    SVC(probability=True, class_weight="balanced"),
    param_grid, cv=3, n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
classifier = grid.best_estimator_
print("Best params:", grid.best_params_)

# ==========================================
# 7. التقييم
# ==========================================
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 45)
print(" System Evaluation Results (LFW Dataset) ")
print("=" * 45)
print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall   : {rec * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")
print("=" * 45)
print(classification_report(y_test, y_pred, target_names=names, zero_division=0))