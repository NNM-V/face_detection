import tkinter as tk
import numpy as np
import cv2 as cv 
from PIL import Image, ImageTk

Cascade_file = "/opt/anaconda3/lib/python3.13/site-packages/cv2/data/"
Face = "haarcascade_frontalface_default.xml"

class FaceRecognition:
    def __init__(self,root):
        self.root = root
        self.root.title('face recognition')
        self.root.geometry('800x600')
        self.create_widgets()
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.root.protocol('WM_DELETE_WINDOW', self.close_window)

    def create_widgets(self):
        self.label = tk.Label(self.root)
        self.label.pack()
        button = tk.Button(text='start', command=self.start_capture)
        button.pack()
        
    def start_capture(self):
        ret, frame = self.cap.read()
        if not ret:
            print('cant recive frame')
            return

        face_img = self.detect_face(frame)
        if face_img is None:
            face_img = frame

        img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.label.configure(image=self.photo)
        self.label.after(10,self.start_capture)

    def detect_face(self,frame):
        face_cascade = cv.CascadeClassifier(Cascade_file+Faces)
        if face_cascade.empty():
            print("カスケードファイルが読み込めません")
            return frame

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        face_rect = face_cascade.detectMultiScale(gray, minSize=(50, 50))
        if len(face_rect) > 0:
            for (x,y,w,h) in face_rect:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),10)

            return frame

    def close_window(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognition(root)
    root.mainloop()