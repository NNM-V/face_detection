import tkinter as tk
import numpy as np
import cv2 as cv 
from PIL import Image, ImageTk

Cascade_file = "/opt/anaconda3/lib/python3.13/site-packages/cv2/data/"
Face = "haarcascade_frontalface_default.xml"
Side = "haarcascade_profileface.xml"
Eyes = "haarcascade_eye_tree_eyeglasses.xml"
Smile = "haarcascade_smile.xml"

class FaceRecognition:
    def __init__(self,root):
        self.root = root
        self.root.title('face recognition')
        self.root.geometry('1200x900')
        self.cam_on = False
        self.cap = None
        self.detect = None
        self.face_cascade = cv.CascadeClassifier(Cascade_file+Face)
        self.faceside_cascade = cv.CascadeClassifier(Cascade_file+Side)
        self.eyes_cascade = cv.CascadeClassifier(Cascade_file+Eyes)
        self.smile_cascade = cv.CascadeClassifier(Cascade_file+Smile)
        self.create_widgets()
        self.root.protocol('WM_DELETE_WINDOW', self.close_window)

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand = True)
        
        self.video_frame = tk.Frame(main_frame, relief='groove', bd=5, width=800, height=600)
        self.video_frame.pack(side='left',padx=10, pady=70)
        self.video_frame.pack_propagate(False)
        self.videolabel = tk.Label(self.video_frame)
        self.videolabel.pack()
        start_button = tk.Button(self.video_frame, text='start camera', width=10, command=self.set_capture)
        start_button.pack(padx=20,pady=80,expand=True)

        command_frame = tk.Frame(main_frame, relief='groove', bd=5, width=270, height=600)
        command_frame.pack(side='left', padx=5, pady=70, fill='y', expand=True)
        command_frame.pack_propagate(False)
        
        cvbutton_frame = tk.Frame(command_frame, relief='groove', bd=5, width=270, height=300)
        cvbutton_frame.pack(side='top', pady=5)
        cvbutton_frame.pack_propagate(False) 
        face_button = tk.Button(cvbutton_frame, text='Face', width=9, height=7,command=lambda: self.set_detectmode('face'))
        face_button.grid(column=0,row=0, padx=1, pady=5)
        face_sidebutton =  tk.Button(cvbutton_frame, text='Face(side)', width=9, height=7,command=lambda: self.set_detectmode('face_side'))
        face_sidebutton.grid(column=1,row=0, padx=1, pady=5)
        eyes_button = tk.Button(cvbutton_frame, text='Eyes', width=9, height=7, command=lambda: self.set_detectmode('eyes'))
        eyes_button.grid(column=0,row=1, padx=1, pady=5)
        smile = tk.Button(cvbutton_frame, text='Smile', width=9, height=7, command=lambda: self.set_detectmode('smile'))
        smile.grid(column=1,row=1, padx=1, pady=5)

        stop_button = tk.Button(command_frame, text='close camera', width=10, command=self.close_camera)
        stop_button.pack(side='top', pady=5)

    def set_detectmode(self,mode):
        self.detect = mode
        
    def set_capture(self):
        self.cam_on=True
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.start_capture()

    def start_capture(self): 
        if self.cam_on:
            ret, frame = self.cap.read()
            if not ret:
                print('cant recive frame')
                return

            width = self.video_frame.winfo_width()
            height = self.video_frame.winfo_height()
            frame = cv.resize(frame, (width, height))
            
            if self.detect=='face':
                detect_img = self.set_facedetect(frame, self.face_cascade)
            elif self.detect == 'face_side':
                detect_img = self.set_facedetect(frame, self.faceside_cascade)
            elif self.detect=='eyes':
                detect_img = self.set_partsdetect(frame, self.face_cascade, self.eyes_cascade)
            elif self.detect=='smile':
                detect_img = self.set_partsdetect(frame, self.face_cascade, self.smile_cascade)
            elif self.detect is None:
                detect_img = frame
          
            img = cv.cvtColor(detect_img, cv.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.videolabel.configure(image=self.photo)
            self.videolabel.after(10,self.start_capture)

    def set_facedetect(self,frame,cascade):
        if cascade.empty():
            print("カスケードファイルが読み込めません")
            return frame

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        if self.detect=='face':
            detect_rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12, minSize=(50, 50))
            for (x,y,w,h) in detect_rect:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),10)
        elif self.detect=='face_side':
            detect_rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (x2,y2,w2,h2) in detect_rect:
                cv.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),10)
        return frame

    def set_partsdetect(self, frame, face_cascade, cascade):
        if face_cascade.empty() or cascade.empty():
            print("カスケードファイルが読み込めません")
            return frame

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        detect_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=15, minSize=(50, 50))
        
        for (x,y,w,h) in detect_rect:
            if self.detect=='eyes':
                eyes = gray[y:int(y+h/2),x:x+w]
                eyes = cv.equalizeHist(eyes)
                eyes_rect = cascade.detectMultiScale(eyes, scaleFactor=1.1, minNeighbors=15, minSize=(15, 15))
                for (x2,y2,w2,h2) in eyes_rect:
                    cv.rectangle(frame,(x+x2,y+y2),(x+x2+w2,y+y2+h2),(0,0,255), 5)
            elif self.detect=='smile':
                mouth = gray[int(y+h/2):(y+h),x:x+w]
                smile_rect = cascade.detectMultiScale(mouth, scaleFactor=1.2, minNeighbors=20, minSize=(15, 15))
                for (x2,y2,w2,h2) in smile_rect:
                    cv.rectangle(frame,(x+x2,int(y+h/2)+y2),(x+x2+w2,int(y+h/2)+y2+h2),(0,0,255), 2)
        return frame

    def close_camera(self):
        if self.cap:
            self.cap.release()
        self.cam_on=False
        self.detect=None
        self.videolabel.config(image='')

    def close_window(self):
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognition(root)
    root.mainloop()