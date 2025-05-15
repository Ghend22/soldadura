import sys
import cv2
import torch
import os
import firebase_admin
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QWidget, QPushButton)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from firebase_admin import credentials, db
from datetime import datetime

# Carga las credenciales del archivo JSON descargado
cred = credentials.Certificate("serviceAccountKey.json")

# Inicializa la aplicación con privilegios de administrador y la URL de tu base de datos
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://eppsoldadura-default-rtdb.firebaseio.com/'
})

# Solución para Windows Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Detección de Soldadura")
        self.setGeometry(100, 100, 800, 600)
        
        # Variables de configuración
        self.video_size = (640, 480)  
        self.model_path = os.path.abspath('best.pt').replace('\\', '/')
        self.video_path = os.path.abspath('solda.mp4').replace('\\', '/')
        
        # Interfaz
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        
        # Etiqueta para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # Botones
        self.btn_start = QPushButton("Iniciar Detección")
        self.btn_start.clicked.connect(self.start_detection)
        self.layout.addWidget(self.btn_start)
        
        self.central_widget.setLayout(self.layout)
        
        # Inicializar modelo y video
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Cargar modelo al iniciar
        self.load_model()
    
    def load_model(self):
        """Carga el modelo YOLOv5"""
        try:
            torch.hub.set_dir(os.getcwd())
            self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', 
                                       path=self.model_path,
                                       force_reload=True,
                                       _verbose=False)
            self.model.eval()
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
    
    def start_detection(self):
        """Inicia el procesamiento del video"""
        if not os.path.exists(self.video_path):
            print(f"❌ Video no encontrado: {self.video_path}")
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("❌ No se pudo abrir el video")
            return
        
        self.btn_start.setEnabled(False)
        self.timer.start(30)  # ~30 FPS
    
    def update_frame(self):
        """Procesa cada frame del video"""
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.btn_start.setEnabled(True)
            return
        
        # Redimensionar frame
        frame = cv2.resize(frame, self.video_size)
        
        # Detección
        results = self.model(frame)
        detections = results.pred[0]

        labels = ['Persona', 'Casco', 'Arco']


        # Registrar resultados en Firebase
        for *box, conf, cls in detections:
            label = labels[int(cls)]
            confidence = float(conf)

        # REGISTRO EN FIREBASE
        ref = db.reference('detecciones')
        ref.push({
            'label': label,
            'confianza': round(confidence, 2),
            'timestamp': datetime.now().isoformat()
        })
        
        # Mostrar resultado en interfaz
        frame_out = results.render()[0]
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_out.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_out.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convertir a formato para Qt
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_out.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_out.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Mostrar en la interfaz
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def closeEvent(self, event):
        """Limpiar al cerrar la aplicación"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())