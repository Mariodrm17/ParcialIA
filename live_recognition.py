#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA EN TIEMPO REAL - RECONOCIMIENTO DE CARTAS
Calibración y reconocimiento EN VIVO desde cámara/ivcam
VERSIÓN CON ROTACIONES - Reconoce cartas en cualquier orientación
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json


class LiveCardRecognition:
    """Sistema de reconocimiento EN TIEMPO REAL."""
    
    def __init__(self, template_dir="templates"):
        self.template_dir = template_dir
        self.template_size = (226, 314)
        self.templates = {}
        
        # Parámetros HSV (se ajustan en vivo)
        self.h_min = 35
        self.h_max = 85
        self.s_min = 40
        self.s_max = 255
        self.v_min = 40
        self.v_max = 255
        
        # Detección
        self.min_area = 10000
        self.max_area = 200000
        self.threshold = 0.65
        
        # Modo
        self.calibration_mode = True
        
        self._load_templates()
        self._load_calibration()
    
    def _load_templates(self):
        """Carga templates."""
        suit_map = {'picas': 'picas', 'corazones': 'corazones', 'diamantes': 'diamantes', 'treboles': 'treboles'}
        
        for suit, code in suit_map.items():
            path = os.path.join(self.template_dir, suit)
            if not os.path.exists(path):
                continue
            
            for img_file in Path(path).glob("*.*"):
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                value = img_file.stem.split('_')[0].upper().replace('AS', 'A')
                name = f"{value}{code}"
                
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                img = cv2.resize(img, self.template_size, interpolation=cv2.INTER_CUBIC)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                self.templates[name] = img
        
        print(f"✅ {len(self.templates)} templates cargados")
    
    def _load_calibration(self):
        """Carga calibración guardada."""
        if os.path.exists('calibration.json'):
            try:
                with open('calibration.json', 'r') as f:
                    data = json.load(f)
                self.h_min = data['h_min']
                self.h_max = data['h_max']
                self.s_min = data['s_min']
                self.s_max = data['s_max']
                self.v_min = data['v_min']
                self.v_max = data['v_max']
                print("✅ Calibración cargada")
            except:
                print("⚠️  Error cargando calibración, usando valores por defecto")
    
    def _save_calibration(self):
        """Guarda calibración."""
        with open('calibration.json', 'w') as f:
            json.dump({
                'h_min': self.h_min, 'h_max': self.h_max,
                's_min': self.s_min, 's_max': self.s_max,
                'v_min': self.v_min, 'v_max': self.v_max
            }, f, indent=4)
        print("💾 Calibración guardada en calibration.json")
    
    def _create_mask(self, frame):
        """Crea máscara del tapete verde."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def _detect_cards(self, frame, mask):
        """Detecta cartas en el frame."""
        card_mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            w, h = int(rect[1][0]), int(rect[1][1])
            if w == 0 or h == 0:
                continue
            
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if not (1.2 < aspect < 1.7):
                continue
            
            card_img = self._extract_card(frame, box, w, h)
            if card_img is not None:
                cards.append((card_img, box))
        
        return cards
    
    def _extract_card(self, frame, box, w, h):
        """Extrae carta."""
        if w > h:
            w, h = h, w
        
        pts = box.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (w, h))
    
    def _recognize_card(self, card_img):
        """Reconoce carta en cualquier orientación (0°, 90°, 180°, 270°)."""
        if len(self.templates) == 0:
            return "?", 0.0
        
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY) if len(card_img.shape) == 3 else card_img
        
        best_score = 0
        best_name = "?"
        best_angle = 0
        
        # Probar 4 rotaciones: 0°, 90°, 180°, 270°
        for angle in [0, 90, 180, 270]:
            # Rotar la imagen de la carta
            if angle == 90:
                rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(gray, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated = gray
            
            # Redimensionar y normalizar
            rotated = cv2.resize(rotated, self.template_size, interpolation=cv2.INTER_CUBIC)
            rotated = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
            
            # Comparar con todos los templates
            for name, template in self.templates.items():
                corr = cv2.matchTemplate(rotated, template, cv2.TM_CCOEFF_NORMED)[0, 0]
                diff = 1.0 - (np.mean(cv2.absdiff(rotated, template)) / 255.0)
                score = 0.6 * corr + 0.4 * diff
                
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_angle = angle
        
        return best_name, best_score
    
    def run_live(self, camera_index=0):
        """Ejecuta sistema EN TIEMPO REAL."""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            print("   Prueba con camera_index=1 o 2 para ivcam")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║       🎴  RECONOCIMIENTO EN TIEMPO REAL + ROTACIONES  🎴    ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("CONTROLES PRINCIPALES:")
        print("  c = MODO CALIBRACIÓN")
        print("  r = MODO RECONOCIMIENTO")
        print("  g = GUARDAR calibración")
        print("  q = SALIR")
        print("\nCALIBRACIÓN (solo en modo calibración):")
        print("  H = Aumentar H Max")
        print("  h = Disminuir H Min")
        print("  J = Aumentar S Max")
        print("  j = Disminuir S Min")
        print("  K = Aumentar V Max")
        print("  k = Disminuir V Min")
        print("  + = Aumentar todos los Max")
        print("  - = Disminuir todos los Min")
        print("\n" + "="*64 + "\n")
        
        cv2.namedWindow('Live Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Recognition', 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crear máscara
            mask = self._create_mask(frame)
            
            # Vista según modo
            if self.calibration_mode:
                # MODO CALIBRACIÓN - Mostrar máscara
                display = self._draw_calibration_view(frame, mask)
            else:
                # MODO RECONOCIMIENTO - Detectar y reconocer
                cards = self._detect_cards(frame, mask)
                display = frame.copy()
                
                for card_img, box in cards:
                    name, score = self._recognize_card(card_img)
                    
                    color = (0, 255, 0) if score >= self.threshold else (0, 165, 255)
                    cv2.drawContours(display, [box], 0, color, 3)
                    
                    label = f"{name}: {score:.2f}"
                    cv2.putText(display, label, tuple(box[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Info en pantalla
                cv2.putText(display, f"MODO: RECONOCIMIENTO | Cartas: {len(cards)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Presiona 'c' para CALIBRAR | 'q' para SALIR", 
                           (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Live Recognition', display)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibration_mode = True
                print("→ MODO: CALIBRACIÓN")
            elif key == ord('r'):
                self.calibration_mode = False
                print("→ MODO: RECONOCIMIENTO")
            elif key == ord('g'):
                self._save_calibration()
            elif self.calibration_mode:
                self._handle_calibration_key(key)
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Finalizado\n")
    
    def _draw_calibration_view(self, frame, mask):
        """Dibuja vista de calibración."""
        h, w = frame.shape[:2]
        
        # Redimensionar para vista dividida
        frame_small = cv2.resize(frame, (w//2, h//2))
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_colored, (w//2, h//2))
        
        result = cv2.bitwise_and(frame, frame, mask=mask)
        result_small = cv2.resize(result, (w//2, h//2))
        
        # Combinar vistas
        top_row = np.hstack([frame_small, mask_small])
        bottom_row = np.hstack([result_small, np.zeros((h//2, w//2, 3), dtype=np.uint8)])
        display = np.vstack([top_row, bottom_row])
        
        # Textos
        cv2.putText(display, "MODO: CALIBRACION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "Original | Mascara", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Resultado | Controles", (10, h//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Valores actuales
        y_start = h//2 + 80
        cv2.putText(display, f"H: [{self.h_min:3d} - {self.h_max:3d}]", 
                   (w//2 + 10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"S: [{self.s_min:3d} - {self.s_max:3d}]", 
                   (w//2 + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"V: [{self.v_min:3d} - {self.v_max:3d}]", 
                   (w//2 + 10, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display, "H/h | J/j | K/k | +/-", 
                   (w//2 + 10, y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, "g=GUARDAR | r=RECONOCER | q=SALIR", 
                   (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display
    
    def _handle_calibration_key(self, key):
        """Maneja teclas de calibración."""
        step = 5
        
        # H Min/Max
        if key == ord('H'):
            self.h_max = min(179, self.h_max + step)
            print(f"H Max: {self.h_max}")
        elif key == ord('h'):
            self.h_min = max(0, self.h_min - step)
            print(f"H Min: {self.h_min}")
        
        # S Min/Max
        elif key == ord('J'):
            self.s_max = min(255, self.s_max + step)
            print(f"S Max: {self.s_max}")
        elif key == ord('j'):
            self.s_min = max(0, self.s_min - step)
            print(f"S Min: {self.s_min}")
        
        # V Min/Max
        elif key == ord('K'):
            self.v_max = min(255, self.v_max + step)
            print(f"V Max: {self.v_max}")
        elif key == ord('k'):
            self.v_min = max(0, self.v_min - step)
            print(f"V Min: {self.v_min}")
        
        # Todos a la vez
        elif key == ord('+') or key == ord('='):
            self.h_max = min(179, self.h_max + step)
            self.s_max = min(255, self.s_max + step)
            self.v_max = min(255, self.v_max + step)
            print(f"Todos Max aumentados")
        elif key == ord('-'):
            self.h_min = max(0, self.h_min - step)
            self.s_min = max(0, self.s_min - step)
            self.v_min = max(0, self.v_min - step)
            print(f"Todos Min disminuidos")


def main():
    """Ejecuta el sistema."""
    
    print("\n🎴 SISTEMA DE RECONOCIMIENTO EN TIEMPO REAL + ROTACIONES\n")
    print("¿Qué cámara estás usando?")
    print("0 = Cámara integrada")
    print("1 = ivcam / cámara externa (prueba primero)")
    print("2 = Otra cámara")
    
    cam_input = input("\nCámara (0/1/2): ").strip()
    camera_index = int(cam_input) if cam_input.isdigit() else 1
    
    system = LiveCardRecognition(template_dir="templates")
    
    print(f"\n🎥 Abriendo cámara {camera_index}...")
    print("⏳ Esperando conexión...\n")
    
    system.run_live(camera_index=camera_index)


if __name__ == "__main__":
    main()