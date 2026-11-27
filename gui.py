import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

PLACEHOLDER = "Write the name here..."

# الألوان
BG = "#1e2a38"
PANEL = "#26364a"
FG = "#ecf0f1"
MUTED = "#95a5a6"
GREEN = "#27ae60"
RED = "#c0392b"
GRAY = "#7f8c8d"

RECOGNIZE_EVERY = 3   # نحسب البصمة كل 3 إطارات فقط (Haar سريع، FaceNet هو الثقيل)
VIDEO_WIDTH = 640     # عرض الفيديو في النافذة
VIDEO_HEIGHT = 480
PANEL_WIDTH = 330     # عرض اللوحة الجانبية (ثابت)
MISS_TOLERANCE = 4    # عدد الإطارات التي نتحمل فيها فشل الكشف قبل إعلان "No face"
SMOOTH = 0.4          # وزن القيمة الجديدة في تنعيم المربع (أقل = أنعم)


class FaceApp:
    def __init__(self, detector, embedder, matcher, database):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.database = database

        # آخر وجه/بصمة في الإطار الحالي فقط (تُصفَّر عند اختفاء الوجه)
        self.last_processed_face = None
        self.last_embedding = None
        self._cached = None          # (identity, distance) لآخر حساب
        self._frame_count = 0
        self._shown_result = None    # لتجنّب تحديث الملصقات بلا داعٍ
        self._list_names = []
        self._after_id = None
        self._box = None             # المربع المنعَّم للرسم
        self._miss = 0               # إطارات متتالية بلا كشف

        # فشل فتح الكاميرا: نرمي خطأ واضحًا قبل إنشاء النافذة
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError("Could not open the camera (index 0). "
                               "Check that it is connected and not used by another app.")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.root = tk.Tk()
        self.root.title("Smart Facial Recognition System")
        self.root.configure(bg=BG)
        # حجم ثابت: تغيّر النصوص في اللوحة اليمنى لا يجب أن يحرّك الفيديو
        self.root.geometry("1010x660")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_layout()
        self._refresh_list()
        self._notify("Ready. Stand in front of the camera.", MUTED)

    # ------------------------------------------------------------------ واجهة
    def _build_layout(self):
        left = tk.Frame(self.root, bg="black", width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        left.pack(side=tk.LEFT, anchor="n", padx=15, pady=15)
        left.pack_propagate(False)
        right = tk.Frame(self.root, bg=BG, width=PANEL_WIDTH)
        right.pack(side=tk.LEFT, fill=tk.Y, anchor="n", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        # --- الفيديو ---
        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack(expand=True)

        # --- بطاقة النتيجة ---
        card = tk.Frame(right, bg=PANEL, padx=15, pady=12, height=125)
        card.pack(fill=tk.X)
        card.pack_propagate(False)
        tk.Label(card, text="RESULT", bg=PANEL, fg=MUTED,
                 font=("Arial", 9, "bold")).pack(anchor="w")
        self.result_title = tk.Label(card, text="No face", bg=PANEL, fg=GRAY,
                                     font=("Arial", 18, "bold"))
        self.result_title.pack(anchor="w")
        self.result_detail = tk.Label(card, text="", bg=PANEL, fg=FG,
                                      font=("Arial", 11), justify="left")
        self.result_detail.pack(anchor="w")

        # --- العتبة ---
        thr = tk.Frame(right, bg=PANEL, padx=15, pady=10)
        thr.pack(fill=tk.X, pady=(10, 0))
        tk.Label(thr, text="THRESHOLD  (lower = stricter)", bg=PANEL, fg=MUTED,
                 font=("Arial", 9, "bold")).pack(anchor="w")
        self.thr_scale = tk.Scale(thr, from_=0.20, to=0.70, resolution=0.01,
                                  orient=tk.HORIZONTAL, bg=PANEL, fg=FG,
                                  highlightthickness=0, troughcolor=BG,
                                  command=self._on_threshold)
        self.thr_scale.set(self.matcher.threshold)
        self.thr_scale.pack(fill=tk.X)

        # --- التسجيل ---
        enroll = tk.Frame(right, bg=PANEL, padx=15, pady=10)
        enroll.pack(fill=tk.X, pady=(10, 0))
        tk.Label(enroll, text="ENROLL", bg=PANEL, fg=MUTED,
                 font=("Arial", 9, "bold")).pack(anchor="w")

        self.name_entry = tk.Entry(enroll, font=("Arial", 14), justify="center", fg=MUTED)
        self.name_entry.insert(0, PLACEHOLDER)
        self.name_entry.pack(fill=tk.X, pady=(4, 6))
        self.name_entry.bind("<FocusIn>", self.clear_placeholder)
        self.name_entry.bind("<FocusOut>", self.restore_placeholder)
        self.name_entry.bind("<Return>", lambda e: self.save_new_person())

        tk.Button(enroll, text="Save the new person", font=("Arial", 12, "bold"),
                  bg=GREEN, fg="white", activebackground="#2ecc71", relief=tk.FLAT,
                  command=self.save_new_person).pack(fill=tk.X)
        tk.Label(enroll, text="Press 3-5 times with slightly different angles.",
                 bg=PANEL, fg=MUTED, font=("Arial", 9)).pack(anchor="w", pady=(4, 0))

        # --- قائمة المسجلين + الحذف ---
        reg = tk.Frame(right, bg=PANEL, padx=15, pady=10)
        reg.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.list_title = tk.Label(reg, text="ENROLLED", bg=PANEL, fg=MUTED,
                                   font=("Arial", 9, "bold"))
        self.list_title.pack(anchor="w")

        list_frame = tk.Frame(reg, bg=PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(list_frame, font=("Arial", 11), height=6,
                                  bg=BG, fg=FG, selectbackground=GREEN,
                                  highlightthickness=0, exportselection=False,
                                  yscrollcommand=scroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)

        tk.Button(reg, text="Delete selected", font=("Arial", 11, "bold"),
                  bg=RED, fg="white", activebackground="#e74c3c", relief=tk.FLAT,
                  command=self.delete_selected).pack(fill=tk.X)

        # --- شريط الرسائل ---
        self.msg_label = tk.Label(right, text="", bg=BG, fg=MUTED, height=3,
                                  font=("Arial", 10), wraplength=PANEL_WIDTH - 20,
                                  justify="left", anchor="nw")
        self.msg_label.pack(fill=tk.X, pady=(10, 0))

    # --------------------------------------------------------------- مساعدات
    def _notify(self, text, color=MUTED):
        self.msg_label.config(text=text, fg=color)

    def _refresh_list(self):
        self._list_names = sorted(self.database.known_embeddings)
        self.listbox.delete(0, tk.END)
        for n in self._list_names:
            self.listbox.insert(tk.END, f"{n}  ({len(self.database.samples[n])})")
        self.list_title.config(text=f"ENROLLED  ({len(self._list_names)})")

    def _on_threshold(self, value):
        self.matcher.threshold = float(value)
        self._cached = None   # يعاد الحساب فورًا بالعتبة الجديدة

    def _set_result(self, title, color, detail=""):
        state = (title, color, detail)
        if state != self._shown_result:      # لا نحدّث الملصق إلا عند التغيّر
            self._shown_result = state
            self.result_title.config(text=title, fg=color)
            self.result_detail.config(text=detail)

    def _reset_face(self):
        self.last_processed_face = None
        self.last_embedding = None
        self._cached = None
        self._box = None

    @staticmethod
    def _smooth_box(prev, new):
        """تنعيم حركة المربع (متوسط متحرك). إذا قفز المربع بعيدًا (وجه آخر) لا ننعّم."""
        new = tuple(float(v) for v in new)
        if prev is None:
            return new
        px, py, pw, ph = prev
        x, y, w, h = new
        if abs((x + w / 2) - (px + pw / 2)) > pw:
            return new
        return tuple(p * (1 - SMOOTH) + n * SMOOTH for p, n in zip(prev, new))

    # ------------------------------------------------------------- الأحداث
    def clear_placeholder(self, event):
        if self.name_entry.get() == PLACEHOLDER:
            self.name_entry.delete(0, tk.END)
            self.name_entry.config(fg="black")

    def restore_placeholder(self, event):
        if not self.name_entry.get().strip():
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, PLACEHOLDER)
            self.name_entry.config(fg=MUTED)

    def save_new_person(self):
        name = self.name_entry.get().strip()

        if not name or name == PLACEHOLDER:
            self._notify("Please write the name first.", "#f39c12")
            return

        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self._notify("Could not read a new camera frame.", "#e74c3c")
                return

            frame = cv2.flip(frame, 1)
            boxes = self.detector.detect(frame)

            if len(boxes) != 1:
                self._notify(
                    "Make sure exactly one face is visible before saving.",
                    "#f39c12",
                )
                return

            face = self.embedder.crop(frame, boxes[0])
            if face is None or face.size == 0:
                self._notify("Could not crop the detected face.", "#e74c3c")
                return

            face_rgb = self.embedder.preprocess(face)
            embedding = self.embedder.embed(face)
            count = self.database.save_person(name, face_rgb, embedding)

        except Exception as error:
            self._notify(f"Could not save: {error}", "#e74c3c")
            return

        self._notify(
            f"Saved [{name}]. Samples: {count}. "
            "Press again with a different angle.",
            "#2ecc71",
        )
        self._refresh_list()
        self._cached = None

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            self._notify("Select a person from the list first.", "#f39c12")
            return
        name = self._list_names[sel[0]]
        if not messagebox.askyesno(
                "Confirm",
                f"Delete [{name}] and their saved face images?\nThis cannot be undone."):
            return
        try:
            removed = self.database.delete_person(name)
        except Exception as e:
            self._notify(f"Could not delete: {e}", "#e74c3c")
            return
        self._notify(f"[{name}] deleted ({removed} image(s) removed).", "#2ecc71")
        self._refresh_list()
        self._cached = None

    # ---------------------------------------------------------------- الرسم
    @staticmethod
    def _draw_label(frame, box, text, color):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        top = max(0, y - th - base - 8)
        cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + base + 8), color, -1)
        cv2.putText(frame, text, (x + 5, top + th + 3), font, scale, (255, 255, 255), thick)

    def update_frame(self):
        try:
            ret, frame = self.camera.read()

            if not ret or frame is None:
                self._reset_face()
                self._set_result("Camera error", RED, "Frame not available")
            else:
                frame = cv2.flip(frame, 1)
                self._frame_count += 1
                raw = self.detector.detect_largest(frame)   # أكبر وجه فقط

                if raw is not None:
                    self._miss = 0
                    self._box = self._smooth_box(self._box, raw)
                else:
                    # Haar يفشل أحيانًا في إطار منفرد؛ نتحمّل بضعة إطارات قبل "No face"
                    self._miss += 1
                    if self._miss > MISS_TOLERANCE:
                        self._reset_face()

                if self._box is None:
                    self._set_result("No face", GRAY, "")
                else:
                    box = tuple(int(round(v)) for v in self._box)   # للرسم فقط
                    # FaceNet كل N إطار على المربع الخام (غير المنعَّم)
                    if raw is not None and (self._cached is None
                                            or self._frame_count % RECOGNIZE_EVERY == 0):
                        face_roi = self.embedder.crop(frame, raw)
                        if face_roi is not None and face_roi.size > 0:
                            resized_face = self.embedder.preprocess(face_roi)
                            live_embedding = self.embedder.embed(face_roi)
                            # إذا لم يُسجَّل أحد بعد يرجع المطابق Unknown بدون خطأ
                            self._cached = self.matcher.recognize(
                                live_embedding, self.database.known_embeddings)
                            self.last_processed_face = resized_face
                            self.last_embedding = live_embedding

                    if self._cached is not None:
                        identity, distance = self._cached
                        if identity != "Unknown":
                            color_bgr, color_hex = (0, 200, 0), GREEN
                            title, text = "Access Granted", f"{identity} ({distance:.2f})"
                            detail = f"{identity}\ndistance {distance:.2f}"
                        else:
                            color_bgr, color_hex = (0, 0, 220), RED
                            title, text = "Access Denied", f"Unknown ({distance:.2f})"
                            detail = f"Unknown\ndistance {distance:.2f}"
                        self._draw_label(frame, box, text, color_bgr)
                        self._set_result(title, color_hex, detail)

                # تصغير/تكبير للعرض فقط (الإحداثيات رُسمت على الإطار الأصلي)
                h, w = frame.shape[:2]
                if w != VIDEO_WIDTH:
                    frame = cv2.resize(frame, (VIDEO_WIDTH, int(h * VIDEO_WIDTH / w)))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)
        except Exception as e:
            # خطأ في إطار واحد لا يجب أن يوقف الحلقة كلها
            print(f"Frame error: {e}")
        finally:
            self._after_id = self.root.after(10, self.update_frame)

    def on_close(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()

    def run(self):
        self.update_frame()
        self.root.mainloop()
        self.camera.release()
        cv2.destroyAllWindows()
