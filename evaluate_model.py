import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet

# ==========================================
# 1. تهيئة نموذج FaceNet (العقل المدبر)
# ==========================================
print("Loading FaceNet model...")
embedder = FaceNet()
print("Model loaded successfully.")

def recognize_face(live_vector, db, threshold=0.75):
    """
    دالة المقارنة: تحسب المسافة (Cosine Distance) بين الصورة الحالية والوجوه في الذاكرة
    """
    min_dist = float('inf')
    identity = "Unknown"
    for name, saved_vector in db.items():
        dist = cosine(live_vector, saved_vector)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    # إذا كانت المسافة أكبر من حد السماحية، يتم رفض الوصول
    if min_dist > threshold:
        identity = "Unknown"
    
    return identity, min_dist

# ==========================================
# 2. تحميل قاعدة البيانات القياسية (LFW)
# ==========================================
print("\nDownloading / Loading LFW Dataset...")
print("This might take a moment if it's the first time...")
# نحضر فقط الأشخاص الذين لديهم 20 صورة على الأقل لضمان اختبار دقيق
lfw_dataset = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0)

images = lfw_dataset.images
target_labels = lfw_dataset.target
target_names = lfw_dataset.target_names

print(f"Total images loaded: {len(images)}")
print(f"Total individuals included: {len(target_names)}")

# ==========================================
# 3. تقسيم البيانات: 5 صور للذاكرة لإنشاء "متوسط البصمة" وباقي الصور للاختبار
# ==========================================
print("\nExtracting Features and Splitting Data (5 for Memory, rest for Testing)...")
known_embeddings = {}
test_images = []
y_true = []

# قاموس مؤقت لجمع أول 5 بصمات لكل شخص
person_references = {}

for i, (img, label_idx) in enumerate(zip(images, target_labels)):
    name = target_names[label_idx]
    
    # تحويل الصورة إلى الحجم المناسب لنموذج FaceNet
    resized_face = cv2.resize(img, (160, 160))
    face_batch = np.expand_dims(resized_face, axis=0)
    
    # استخراج بصمة الوجه (Embedding)
    embedding = embedder.embeddings(face_batch)[0]

    if name not in person_references:
        person_references[name] = []
        
    # أخذ أول 5 صور لتكوين الذاكرة المرجعية
    if len(person_references[name]) < 5:
        person_references[name].append(embedding)
    else:
        # باقي الصور نستخدمها لاختبار دقة النظام
        test_images.append(embedding)
        y_true.append(name)
        
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(images)} images...")

# حساب "المتوسط الحسابي" لبصمات الوجوه الخمسة لإنشاء بصمة مرجعية دقيقة جداً
for name, embeddings_list in person_references.items():
    known_embeddings[name] = np.mean(embeddings_list, axis=0)

# ==========================================
# 4. التقييم (Evaluation & Metrics)
# ==========================================
print("\nRunning Model Evaluation...")
y_pred = []
 
for test_embedding in test_images:
    # مقارنة كل صورة اختبار بالذاكرة المرجعية
    identity, dist = recognize_face(test_embedding, known_embeddings, threshold=0.75)
    y_pred.append(identity)

# حساب المقاييس
# يتم استخدام average='weighted' لحساب المتوسط بدقة في حالات التصنيف المتعددة (Multi-class)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# طباعة النتائج النهائية
print("\n" + "="*45)        
print(" 📊 System Evaluation Results (LFW Dataset) ")
print("="*45)
print(f"Accuracy  (Overall accuracy):    {acc * 100:.2f}%")
print(f"Precision (Positive precision):   {prec * 100:.2f}%")
print(f"Recall    (Recovery rate):    {rec * 100:.2f}%")
print(f"F1-Score  (The integrated scale):    {f1 * 100:.2f}%")
print("="*45)