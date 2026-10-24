import cv2
import numpy as np 
import os
import pickle
from scipy.spatial.distance import cosine
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# Invoking the FaceNet model
from keras_facenet import FaceNet

# ==========================================
# 1. تجهيز العقل المدبر (نموذج FaceNet) طارق
# ==========================================
print("Loading the FaceNet model...")
embedder = FaceNet()
print("The model has been successfully loaded.")


# ==========================================
# 2.زينب   الذاكرة (تحميل قاعدة البيانات) 
# ==========================================
db_path = "face_embeddings.pkl"
known_embeddings = {}

if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        loaded_db = pickle.load(f)
        
    # --- التعديل هنا: فلترة البيانات غير المتوافقة ---
    for name, embedding in loaded_db.items():
        # التحقق من أن حجم البصمة هو 512 الخاص بنموذج FaceNet
        if len(embedding) == 512:
            known_embeddings[name] = embedding
        else:
            print(f" Ignored [{name}] because embedding size ({len(embedding)}) is incompatible with current model (512).")
    # ------------------------------------------

    print(f"A person has been loaded {len(known_embeddings)} from memory.")
else:
    print("The database is currently empty.")

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)
image_count = len(os.listdir(dataset_dir)) if os.path.exists(dataset_dir) else 0   


# ==========================================    
# 3.  زكيه   دوال المقارنة واستخراج الملامح   
# ==========================================
def recognize_face(live_vector, db, threshold=0.5):
    if not db:
        return "Unknown", 1.0

    min_dist = float('inf')
    identity = "Unknown"

    for name, saved_vector in db.items():
        dist = cosine(live_vector, saved_vector)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > threshold:
        identity = "Unknown"
    
    return identity, min_dist


# ==========================================
# مهمة أحمد: دمج المرحلة الأولى → تجهيز Input Pipeline → ربط مرحلة Feature Extraction ببقية البرنامج
# ==========================================
# متغيرات عالمية لحفظ آخر وجه تم التقاطه
last_processed_face = None
last_embedding = None


# ==========================================
# طارق: Integration & Camera Loop
# ==========================================

# Definition and download of the Haar Cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
camera = cv2.VideoCapture(0)

# إنشاء نافذة البرنامج
root = tk.Tk()
root.title("Smart Facial Recognition System")
root.geometry("700x650")
root.configure(bg="#2c3e50")

# مكان عرض الكاميرا
video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

# إطار (مربع) يضم مكان إدخال الاسم وزر الحفظ
input_frame = tk.Frame(root, bg="#2c3e50")
input_frame.pack(pady=10)

name_entry = tk.Entry(input_frame, font=("Arial", 16), width=20, justify='center')
name_entry.insert(0, "Write the name here...")
name_entry.pack(side=tk.RIGHT, padx=10)

# دالة تُمسح النص الافتراضي عند الضغط على مربع النص
def clear_placeholder(event):
    if name_entry.get() == "Write the name here...":
        name_entry.delete(0, tk.END)
name_entry.bind("<FocusIn>", clear_placeholder)

# دالة حفظ الوجه الجديد
def save_new_person():
    global image_count, known_embeddings

    name = name_entry.get().strip()

    if name == "" or name == "Write the name here...":
        messagebox.showwarning("Warning", "Please write the name first.")
        return

    if last_processed_face is not None and last_embedding is not None:
        #save Image
        image_count +=1
        file_path = f"{dataset_dir}/{name}_{image_count}.jpg"
        bgr_face = cv2.cvtColor(last_processed_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr_face)

        # جزء من مهمة زينب: تخزين الـ Face Embeddings في قاعدة البيانات
        known_embeddings[name] = last_embedding
        with open(db_path, "wb") as f:
            pickle.dump(known_embeddings, f)

        messagebox.showinfo("Success", f"[{name}] has been successfully added to the database!")
        name_entry.delete(0, tk.END)
    else:
        messagebox.showerror("Error", "Error: Face not detected by the camera!")

# زر الحفظ
save_btn = tk.Button(input_frame, text="Save the new person", font=("Arial", 14, "bold"),
                     bg="#27ae60", fg="white", command=save_new_person)
save_btn.pack(side=tk.LEFT, padx=10)


# ==========================================
# دالة تحديث الكاميرا المستمر (القلب النابض للبرنامج)
# ==========================================
def update_frame():
    global last_processed_face, last_embedding
    
    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            if face_roi is not None and face_roi.size > 0:
                
                # جزء من مهمة أحمد: تجهيز Input Pipeline
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                resized_face = cv2.resize(face_rgb, (160, 160))
                face_batch = np.expand_dims(resized_face, axis=0)

                # جزء من مهمة طارق: تمرير الوجه إلى CNN وإنتاج Feature Vector
                live_embedding = embedder.embeddings(face_batch)[0]
                
                # جزء من مهمة زكية: Feature Comparison وتحديد الـ Identity أو Unknown
                identity, distance = recognize_face(live_embedding, known_embeddings, threshold=0.5)

                # التعديل الجديد: تحديد حالة الوصول واللون بناءً على النتيجة
                if identity != "Unknown":
                    color = (0, 255, 0) #green
                    status_text = f"Access Granted: {identity}"
                else:
                    color = (0, 0, 255) #reed
                    status_text = "Access Denied: Unknown"

                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

                # طباعة حالة الوصول ونسبة الخطأ (Distance) على الشاشة
                cv2.putText(frame, f"{status_text} ({distance:.2f})", (x, y-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                last_processed_face = resized_face
                last_embedding = live_embedding 

        # جزء من مهمة أحمد: ربط النتائج وعرضها في الواجهة
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

camera.release()
cv2.destroyAllWindows()