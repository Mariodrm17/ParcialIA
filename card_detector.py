"""
CARD DETECTOR - Detección de Cartas en el Tapete
Detecta y extrae cartas usando la calibración del tapete verde
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

class CardDetector:
    """Detector de cartas sobre tapete verde calibrado"""
    
    def __init__(self, calibration_file: str = "green_calibration.json"):
        self.calibration_file = Path(calibration_file)
        self.green_lower = None
        self.green_upper = None
        self.is_calibrated = False
        
        # Parámetros de detección
        self.min_area = 2000   # Área mínima de carta
        self.max_area = 50000  # Área máxima de carta
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 0.9
        
        # Cargar calibración automáticamente
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """Carga calibración del tapete verde"""
        if not self.calibration_file.exists():
            print("⚠️  No hay calibración. Ejecuta: python 2_green_calibrator.py")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                config = json.load(f)
            
            self.green_lower = np.array(config['green_lower'])
            self.green_upper = np.array(config['green_upper'])
            self.is_calibrated = True
            
            print("✅ Calibración cargada")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando calibración: {e}")
            return False
    
    def detect_cards(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Detecta cartas en el frame
        
        Args:
            frame: Frame de la cámara (BGR)
            
        Returns:
            (lista_imagenes_cartas, lista_posiciones)
            donde posiciones es (x, y, width, height)
        """
        if not self.is_calibrated:
            return [], []
        
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Aplicar máscara verde
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Morfología para limpiar
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir: cartas = blanco, tapete = negro
            cards_mask = cv2.bitwise_not(green_mask)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(
                cards_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            cards = []
            positions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área
                if not (self.min_area < area < self.max_area):
                    continue
                
                # Aproximar a polígono
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buscar rectángulos (4 vértices)
                if len(approx) != 4:
                    continue
                
                # Obtener rectángulo delimitador
                x, y, w, h = cv2.boundingRect(contour)
                
                # Verificar relación de aspecto (cartas son ~0.6-0.7)
                aspect_ratio = w / h if h > 0 else 0
                
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue
                
                # Extraer carta con margen
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                card_img = frame[y1:y2, x1:x2]
                
                if card_img.size > 0:
                    cards.append(card_img)
                    positions.append((x, y, w, h))
            
            return cards, positions
            
        except Exception as e:
            print(f"❌ Error detectando cartas: {e}")
            return [], []
    
    def draw_detections(self, frame: np.ndarray, 
                       positions: List[Tuple[int, int, int, int]],
                       labels: Optional[List[str]] = None,
                       confidences: Optional[List[float]] = None) -> np.ndarray:
        """
        Dibuja las detecciones en el frame
        
        Args:
            frame: Frame original
            positions: Lista de posiciones (x, y, w, h)
            labels: Lista opcional de etiquetas
            confidences: Lista opcional de confianzas
            
        Returns:
            Frame con detecciones dibujadas
        """
        result = frame.copy()
        
        for i, (x, y, w, h) in enumerate(positions):
            # Color según confianza si disponible
            if confidences and i < len(confidences):
                conf = confidences[i]
                if conf >= 0.8:
                    color = (0, 255, 0)  # Verde
                elif conf >= 0.65:
                    color = (0, 255, 255)  # Amarillo
                elif conf >= 0.5:
                    color = (0, 165, 255)  # Naranja
                else:
                    color = (0, 0, 255)  # Rojo
            else:
                color = (0, 255, 0)  # Verde por defecto
            
            # Dibujar rectángulo
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
            
            # Dibujar etiqueta si disponible
            if labels and i < len(labels):
                label = labels[i]
                
                # Fondo para texto
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(result, (x, y-text_h-10), 
                            (x+text_w+10, y), color, -1)
                
                # Texto
                cv2.putText(result, label, (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Dibujar confianza si disponible
            if confidences and i < len(confidences):
                conf_text = f"{confidences[i]:.1%}"
                cv2.putText(result, conf_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def get_detection_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Obtiene la máscara de detección (útil para debug)
        
        Returns:
            Máscara donde blanco = posibles cartas
        """
        if not self.is_calibrated:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        cards_mask = cv2.bitwise_not(green_mask)
        
        return cards_mask


# Test del módulo
if __name__ == "__main__":
    print("🧪 TEST: Card Detector")
    
    detector = CardDetector()
    
    if not detector.is_calibrated:
        print("❌ No hay calibración. Ejecuta: python 2_green_calibrator.py")
        exit(1)
    
    # Probar con cámara
    cam_idx = input("Índice de cámara (default=0): ").strip()
    cam_idx = int(cam_idx) if cam_idx else 0
    
    cap = cv2.VideoCapture(cam_idx)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir cámara {cam_idx}")
        exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n✅ Test iniciado. Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar cartas
        cards, positions = detector.detect_cards(frame)
        
        # Dibujar
        display = detector.draw_detections(frame, positions)
        
        # Info
        cv2.putText(display, f"Cartas detectadas: {len(cards)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Presiona 'q' para salir",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar
        cv2.imshow("Card Detector Test", display)
        
        # Mostrar máscara (debug)
        mask = detector.get_detection_mask(frame)
        cv2.imshow("Mascara (blanco = posibles cartas)", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completado")