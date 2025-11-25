"""
CARD DETECTOR PROFESIONAL
Detección optimizada de cartas de poker sobre tapete verde
Con calibración perfecta incluida
"""

import cv2
import numpy as np
from typing import List, Tuple
import json
from pathlib import Path

class CardDetectorPro:
    """
    Detector profesional de cartas
    - Calibración automática del tapete verde
    - Detección robusta de contornos
    - Filtrado inteligente de falsas detecciones
    """
    
    def __init__(self):
        # Calibración óptima para tapete verde de poker
        # Estos valores están optimizados para tapetes estándar
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
        # Parámetros de detección
        self.min_area = 3000      # Área mínima carta
        self.max_area = 45000     # Área máxima carta
        self.min_aspect = 0.55    # Relación aspecto mínima
        self.max_aspect = 0.85    # Relación aspecto máxima
        
        self.config_file = Path("poker_calibration.json")
        
        # Intentar cargar calibración personalizada
        self._load_calibration()
    
    def _load_calibration(self):
        """Carga calibración si existe"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.green_lower = np.array(config['green_lower'])
                self.green_upper = np.array(config['green_upper'])
                print("✅ Calibración personalizada cargada")
            except:
                print("⚠️  Usando calibración por defecto")
    
    def calibrate(self, camera_index: int = 0):
        """
        Calibración interactiva del tapete verde
        OPTIMIZADA para tapetes de poker
        """
        print("\n" + "=" * 70)
        print("🎨 CALIBRACIÓN DEL TAPETE VERDE")
        print("=" * 70)
        print("\n📋 INSTRUCCIONES:")
        print("1. Coloca SOLO el tapete verde sin cartas")
        print("2. Buena iluminación uniforme")
        print("3. Ajusta trackbars hasta que:")
        print("   ✅ Tapete = COMPLETAMENTE BLANCO")
        print("   ✅ Resto = NEGRO")
        print("4. 's' = GUARDAR | 'q' = CANCELAR")
        print("=" * 70)
        
        input("\n⏸️  ENTER para continuar...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ No se pudo abrir cámara")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Valores iniciales optimizados
        h_min, h_max = 35, 85
        s_min, s_max = 40, 255
        v_min, v_max = 40, 255
        
        cv2.namedWindow('Calibracion')
        cv2.namedWindow('Resultado')
        
        # Trackbars
        cv2.createTrackbar('H Min', 'Calibracion', h_min, 180, lambda x: None)
        cv2.createTrackbar('H Max', 'Calibracion', h_max, 180, lambda x: None)
        cv2.createTrackbar('S Min', 'Calibracion', s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', 'Calibracion', s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', 'Calibracion', v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', 'Calibracion', v_max, 255, lambda x: None)
        
        print("\n🔧 Ajustando... (ventanas abiertas)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Leer valores
            h_min = cv2.getTrackbarPos('H Min', 'Calibracion')
            h_max = cv2.getTrackbarPos('H Max', 'Calibracion')
            s_min = cv2.getTrackbarPos('S Min', 'Calibracion')
            s_max = cv2.getTrackbarPos('S Max', 'Calibracion')
            v_min = cv2.getTrackbarPos('V Min', 'Calibracion')
            v_max = cv2.getTrackbarPos('V Max', 'Calibracion')
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, 
                              np.array([h_min, s_min, v_min]),
                              np.array([h_max, s_max, v_max]))
            
            # Limpiar
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Porcentaje
            percent = (cv2.countNonZero(mask) / mask.size) * 100
            
            # Mostrar
            info = frame.copy()
            cv2.putText(info, f"Detectado: {percent:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(info, "'s'=Guardar | 'q'=Cancelar", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Calibracion', info)
            cv2.imshow('Resultado', mask)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n❌ Cancelado")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('s'):
                if percent < 15 or percent > 90:
                    print(f"\n⚠️  Detección: {percent:.1f}% - Ajusta mejor")
                    continue
                
                # Guardar
                self.green_lower = np.array([h_min, s_min, v_min])
                self.green_upper = np.array([h_max, s_max, v_max])
                
                config = {
                    'green_lower': self.green_lower.tolist(),
                    'green_upper': self.green_upper.tolist()
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                print("\n✅ CALIBRACIÓN GUARDADA")
                print(f"   H: [{h_min}-{h_max}]")
                print(f"   S: [{s_min}-{s_max}]")
                print(f"   V: [{v_min}-{v_max}]")
                
                cap.release()
                cv2.destroyAllWindows()
                return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def detect_cards(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """
        Detecta cartas en el frame
        
        Returns:
            (lista_imagenes_cartas, lista_posiciones)
        """
        # HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara verde
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morfología agresiva
        kernel = np.ones((7, 7), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Invertir (cartas = blanco)
        cards_mask = cv2.bitwise_not(green_mask)
        
        # Contornos
        contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        positions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtro por área
            if not (self.min_area < area < self.max_area):
                continue
            
            # Aproximar a polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Debe ser rectángulo (4 lados)
            if len(approx) != 4:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Relación de aspecto (cartas ~0.6-0.7)
            aspect = w / h if h > 0 else 0
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            
            # Extraer carta con margen
            margin = 25
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            card_img = frame[y1:y2, x1:x2]
            
            if card_img.size > 0:
                cards.append(card_img)
                positions.append((x, y, w, h))
        
        return cards, positions
    
    def draw_detections(self, frame: np.ndarray, positions: List[Tuple],
                       labels: List[str] = None, 
                       confidences: List[float] = None) -> np.ndarray:
        """Dibuja detecciones en el frame"""
        result = frame.copy()
        
        for i, (x, y, w, h) in enumerate(positions):
            # Color según confianza
            if confidences and i < len(confidences):
                conf = confidences[i]
                if conf >= 0.80:
                    color = (0, 255, 0)
                elif conf >= 0.65:
                    color = (0, 255, 255)
                elif conf >= 0.50:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            # Rectángulo
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
            
            # Etiqueta
            if labels and i < len(labels):
                label = labels[i]
                
                # Fondo para texto
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(result, (x, y-th-10), (x+tw+10, y), color, -1)
                cv2.putText(result, label, (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Confianza
            if confidences and i < len(confidences):
                conf_text = f"{confidences[i]:.0%}"
                cv2.putText(result, conf_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result


# Test rápido
if __name__ == "__main__":
    print("\n🧪 TEST: Card Detector Pro")
    
    detector = CardDetectorPro()
    
    print(f"\n✅ Detector listo")
    print(f"   Verde detectado: H[{detector.green_lower[0]}-{detector.green_upper[0]}]")
    print(f"   Área cartas: {detector.min_area}-{detector.max_area}")
    
    recalibrar = input("\n¿Calibrar tapete? (s/n): ").lower()
    if recalibrar == 's':
        detector.calibrate()