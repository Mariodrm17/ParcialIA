#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Reconocimiento de Cartas - VERSIÓN FINAL
Calibración interactiva + Reconocimiento por templates
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional


class CardRecognitionSystem:
    """Sistema de reconocimiento de cartas con calibración interactiva."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = template_dir
        self.template_size = (226, 314)
        self.templates = {}
        self.card_names = []
        
        # Parámetros de tapete verde
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
        # Parámetros de detección
        self.min_card_area = 10000
        self.max_card_area = 200000
        self.match_threshold = 0.65
        
        # Mapeo de palos
        self.suit_map = {
            'picas': 'S',
            'corazones': 'H',
            'diamantes': 'D',
            'treboles': 'C'
        }
        
        self._load_templates()
    
    def _load_templates(self):
        """Carga templates desde estructura por palos."""
        if not os.path.exists(self.template_dir):
            print(f"❌ No existe '{self.template_dir}/'")
            return
        
        print(f"📚 Cargando templates...\n")
        
        for suit_folder, suit_code in self.suit_map.items():
            suit_path = os.path.join(self.template_dir, suit_folder)
            
            if not os.path.exists(suit_path):
                continue
            
            print(f"🃏 {suit_folder.upper()}...")
            
            for img_path in Path(suit_path).glob("*.*"):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                # Extraer valor (2, 3, AS, K, etc.)
                filename = img_path.stem
                value = filename.split('_')[0].upper() if '_' in filename else filename.upper()
                value = value.replace('AS', 'A')
                
                # Crear nombre: AS, 2C, KH, etc.
                card_name = f"{value}{suit_code}"
                
                # Cargar y procesar
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Redimensionar
                img = cv2.resize(img, self.template_size, interpolation=cv2.INTER_CUBIC)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                
                self.templates[card_name] = img
                self.card_names.append(card_name)
                print(f"   ✓ {card_name}")
        
        print(f"\n✅ {len(self.templates)} templates cargados\n")
    
    def calibrate_interactive(self, image_path: str):
        """Calibración INTERACTIVA del tapete verde."""
        print("\n" + "="*70)
        print("🎯 CALIBRACIÓN INTERACTIVA DEL TAPETE VERDE")
        print("="*70)
        print("\nCONTROLES:")
        print("  • Ajusta sliders H/S/V hasta que SOLO el tapete esté BLANCO")
        print("  • 's' = GUARDAR calibración")
        print("  • 'r' = RESETEAR valores")
        print("  • 'q' = SALIR sin guardar")
        print("="*70 + "\n")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error al cargar: {image_path}")
            return False
        
        window = "Calibracion Tapete Verde"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1200, 700)
        
        def nothing(x): pass
        
        # Crear trackbars
        cv2.createTrackbar('H Min', window, self.green_lower[0], 179, nothing)
        cv2.createTrackbar('H Max', window, self.green_upper[0], 179, nothing)
        cv2.createTrackbar('S Min', window, self.green_lower[1], 255, nothing)
        cv2.createTrackbar('S Max', window, self.green_upper[1], 255, nothing)
        cv2.createTrackbar('V Min', window, self.green_lower[2], 255, nothing)
        cv2.createTrackbar('V Max', window, self.green_upper[2], 255, nothing)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        while True:
            h_min = cv2.getTrackbarPos('H Min', window)
            h_max = cv2.getTrackbarPos('H Max', window)
            s_min = cv2.getTrackbarPos('S Min', window)
            s_max = cv2.getTrackbarPos('S Max', window)
            v_min = cv2.getTrackbarPos('V Min', window)
            v_max = cv2.getTrackbarPos('V Max', window)
            
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            # Crear máscara
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Visualización
            result = cv2.bitwise_and(image, image, mask=mask)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            top = np.hstack([image, mask_bgr])
            bottom = np.hstack([result, np.zeros_like(image)])
            combined = np.vstack([top, bottom])
            
            # Texto
            text = f"HSV: [{h_min}-{h_max}, {s_min}-{s_max}, {v_min}-{v_max}]"
            cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Original | Mascara | Resultado", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(combined, "s=GUARDAR | r=RESET | q=SALIR", (10, combined.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(window, combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.green_lower = lower
                self.green_upper = upper
                print("\n✅ CALIBRACIÓN GUARDADA:")
                print(f"   Lower: {self.green_lower}")
                print(f"   Upper: {self.green_upper}")
                
                with open('calibration.json', 'w') as f:
                    json.dump({
                        'green_lower': self.green_lower.tolist(),
                        'green_upper': self.green_upper.tolist()
                    }, f, indent=4)
                print("💾 Guardado en: calibration.json\n")
                break
            
            elif key == ord('r'):
                cv2.setTrackbarPos('H Min', window, 35)
                cv2.setTrackbarPos('H Max', window, 85)
                cv2.setTrackbarPos('S Min', window, 40)
                cv2.setTrackbarPos('S Max', window, 255)
                cv2.setTrackbarPos('V Min', window, 40)
                cv2.setTrackbarPos('V Max', window, 255)
            
            elif key == ord('q'):
                print("\n⚠️  Cancelado (no guardado)\n")
                break
        
        cv2.destroyAllWindows()
        return True
    
    def load_calibration(self):
        """Carga calibración guardada."""
        if os.path.exists('calibration.json'):
            with open('calibration.json', 'r') as f:
                data = json.load(f)
            self.green_lower = np.array(data['green_lower'])
            self.green_upper = np.array(data['green_upper'])
            print(f"✅ Calibración cargada:")
            print(f"   Lower: {self.green_lower}")
            print(f"   Upper: {self.green_upper}\n")
            return True
        return False
    
    def detect_cards(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detecta cartas en la imagen."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Invertir (cartas = NO verde)
        card_mask = cv2.bitwise_not(mask)
        
        # Contornos
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_card_area or area > self.max_card_area:
                continue
            
            # Rectángulo rotado
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            w, h = int(rect[1][0]), int(rect[1][1])
            aspect = max(w, h) / (min(w, h) + 1e-6)
            
            if not (1.2 < aspect < 1.7):
                continue
            
            # Extraer carta
            card_img = self._extract_card(image, box, w, h)
            if card_img is not None:
                cards.append((card_img, box))
        
        return cards
    
    def _extract_card(self, image, box, w, h):
        """Extrae y endereza la carta."""
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
        return cv2.warpPerspective(image, M, (w, h))
    
    def recognize_card(self, card_img: np.ndarray) -> Tuple[str, float]:
        """Reconoce una carta comparando con templates."""
        if len(self.templates) == 0:
            return "?", 0.0
        
        # Procesar carta
        if len(card_img.shape) == 3:
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_img.copy()
        
        gray = cv2.resize(gray, self.template_size, interpolation=cv2.INTER_CUBIC)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Comparar con templates
        best_score = 0
        best_card = "?"
        
        for card_name, template in self.templates.items():
            # Correlación
            corr = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            # Diferencia
            diff = 1.0 - (np.mean(cv2.absdiff(gray, template)) / 255.0)
            
            # Score final
            score = 0.6 * corr + 0.4 * diff
            
            if score > best_score:
                best_score = score
                best_card = card_name
        
        return best_card, best_score
    
    def process_image(self, image_path: str, output_dir: str = "output"):
        """Procesa imagen completa."""
        print(f"\n{'='*70}")
        print(f"🎴 PROCESANDO: {image_path}")
        print(f"{'='*70}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error al cargar imagen")
            return []
        
        # Detectar
        cards = self.detect_cards(image)
        print(f"🎴 Detectadas: {len(cards)} cartas\n")
        
        if len(cards) == 0:
            print("⚠️  No se detectaron cartas")
            return []
        
        # Reconocer
        results = []
        result_img = image.copy()
        
        for idx, (card_img, box) in enumerate(cards):
            card_name, score = self.recognize_card(card_img)
            
            print(f"Carta {idx+1}: {card_name} (score: {score:.3f})")
            
            # Guardar carta recortada
            cv2.imwrite(f"{output_dir}/card_{idx+1}.png", card_img)
            
            # Dibujar en resultado
            color = (0, 255, 0) if score >= self.match_threshold else (0, 165, 255)
            cv2.drawContours(result_img, [box], 0, color, 3)
            cv2.putText(result_img, f"{card_name}:{score:.2f}", tuple(box[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            results.append({
                'card_number': idx + 1,
                'recognized': card_name,
                'confidence': float(score)
            })
        
        # Guardar resultado
        cv2.imwrite(f"{output_dir}/result.png", result_img)
        
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n💾 Resultados guardados en: {output_dir}/")
        print(f"{'='*70}\n")
        
        return results


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🎴  SISTEMA DE RECONOCIMIENTO DE CARTAS  🎴        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    system = CardRecognitionSystem(template_dir="templates")
    
    if not system.load_calibration():
        print("⚠️  No hay calibración. Usa:")
        print("   system.calibrate_interactive('imagen_tapete.jpg')\n")