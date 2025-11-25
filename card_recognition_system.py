#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Reconocimiento de Cartas de Poker - VERSIÓN MEJORADA
Adaptado para estructura de templates por palos
Calibración interactiva del tapete verde
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


class CardRecognitionSystemV2:
    """
    Sistema mejorado de reconocimiento de cartas.
    Optimizado para estructura de templates organizados por palos.
    """
    
    def __init__(self, template_dir: str = "templates", template_size: Tuple[int, int] = (226, 314)):
        """
        Inicializa el sistema de reconocimiento.
        
        Args:
            template_dir: Directorio raíz con subdirectorios por palos
            template_size: Tamaño estándar de las plantillas (ancho, alto)
        """
        self.template_dir = template_dir
        self.template_size = template_size
        self.templates = {}
        self.card_names = []
        
        # Parámetros de tapete verde (se calibrarán interactivamente)
        self.green_hsv_lower = np.array([35, 40, 40])
        self.green_hsv_upper = np.array([85, 255, 255])
        
        # Parámetros morfológicos
        self.morph_kernel_size = 5
        self.min_card_area = 10000
        self.max_card_area = 200000
        
        # Parámetros de comparación
        self.match_threshold = 0.65
        
        # Nombres de palos
        self.suit_names = {
            'picas': 'S',      # Spades
            'corazones': 'H',   # Hearts  
            'diamantes': 'D',   # Diamonds
            'treboles': 'C'     # Clubs
        }
        
        self._load_templates_from_structure()
    
    def _load_templates_from_structure(self):
        """
        Carga templates desde estructura organizada por palos.
        Estructura esperada:
        templates/
        ├── picas/
        │   ├── 2_PICAS.png
        │   ├── AS_PICAS.png
        │   └── ...
        ├── corazones/
        ├── diamantes/
        └── treboles/
        """
        if not os.path.exists(self.template_dir):
            print(f"⚠️  Directorio '{self.template_dir}' no encontrado.")
            return
        
        print(f"📚 Cargando templates desde '{self.template_dir}'...\n")
        
        # Recorrer cada subdirectorio de palos
        for suit_folder, suit_code in self.suit_names.items():
            suit_path = os.path.join(self.template_dir, suit_folder)
            
            if not os.path.exists(suit_path):
                print(f"⚠️  No se encontró carpeta: {suit_folder}/")
                continue
            
            print(f"🃏 Cargando {suit_folder.upper()}...")
            
            # Buscar archivos PNG en el directorio
            template_files = list(Path(suit_path).glob("*.png")) + \
                           list(Path(suit_path).glob("*.PNG")) + \
                           list(Path(suit_path).glob("*.jpg")) + \
                           list(Path(suit_path).glob("*.JPG"))
            
            for template_path in sorted(template_files):
                # Extraer valor de la carta del nombre del archivo
                # Ejemplos: "2_PICAS.png" → "2", "AS_PICAS.png" → "A"
                filename = template_path.stem
                
                # Extraer el valor (antes del primer _ o espacio)
                if '_' in filename:
                    value = filename.split('_')[0]
                elif ' ' in filename:
                    value = filename.split(' ')[0]
                else:
                    value = filename
                
                # Normalizar el valor
                value = value.upper().replace('AS', 'A').replace('PICAS', '').replace('CORAZONES', '').replace('DIAMANTES', '').replace('TREBOLES', '').strip()
                
                # Crear nombre estándar: ValorPalo (ej: "AS", "2C", "KH")
                card_name = f"{value}{suit_code}"
                
                # Cargar template
                template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                
                if template_img is None:
                    print(f"   ⚠️  Error cargando {template_path.name}")
                    continue
                
                # Redimensionar si es necesario
                if template_img.shape[:2] != (self.template_size[1], self.template_size[0]):
                    template_img = cv2.resize(template_img, self.template_size, 
                                             interpolation=cv2.INTER_CUBIC)
                
                # Normalizar
                template_normalized = cv2.normalize(template_img, None, 0, 255, cv2.NORM_MINMAX)
                
                self.templates[card_name] = template_normalized
                self.card_names.append(card_name)
                print(f"   ✓ {card_name} ← {template_path.name}")
        
        print(f"\n✅ Total de templates cargados: {len(self.templates)}")
        print(f"   Cartas disponibles: {', '.join(sorted(self.card_names))}\n")
    
    def calibrate_interactive(self, image_path: str):
        """
        Calibración INTERACTIVA del tapete verde con trackbars.
        Permite ajustar los rangos HSV en tiempo real.
        
        Args:
            image_path: Ruta a imagen con el tapete verde visible
        """
        print("\n" + "="*70)
        print("🎯 CALIBRACIÓN INTERACTIVA DEL TAPETE VERDE")
        print("="*70)
        print("\nInstrucciones:")
        print("1. Ajusta los sliders hasta que SOLO el tapete verde esté en BLANCO")
        print("2. Las cartas y otros objetos deben estar en NEGRO")
        print("3. Presiona 'q' cuando estés satisfecho")
        print("4. Presiona 'r' para resetear valores")
        print("5. Presiona 's' para guardar y salir")
        print("\n" + "="*70 + "\n")
        
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error al cargar la imagen: {image_path}")
            return False
        
        # Crear ventana
        window_name = "Calibracion Tapete Verde"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 700)
        
        # Función callback (no hace nada, solo necesaria para trackbars)
        def nothing(x):
            pass
        
        # Crear trackbars para HSV
        cv2.createTrackbar('H Min', window_name, self.green_hsv_lower[0], 179, nothing)
        cv2.createTrackbar('H Max', window_name, self.green_hsv_upper[0], 179, nothing)
        cv2.createTrackbar('S Min', window_name, self.green_hsv_lower[1], 255, nothing)
        cv2.createTrackbar('S Max', window_name, self.green_hsv_upper[1], 255, nothing)
        cv2.createTrackbar('V Min', window_name, self.green_hsv_lower[2], 255, nothing)
        cv2.createTrackbar('V Max', window_name, self.green_hsv_upper[2], 255, nothing)
        
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        print("⏳ Ajustando sliders... (Presiona 's' para guardar)\n")
        
        while True:
            # Leer valores de trackbars
            h_min = cv2.getTrackbarPos('H Min', window_name)
            h_max = cv2.getTrackbarPos('H Max', window_name)
            s_min = cv2.getTrackbarPos('S Min', window_name)
            s_max = cv2.getTrackbarPos('S Max', window_name)
            v_min = cv2.getTrackbarPos('V Min', window_name)
            v_max = cv2.getTrackbarPos('V Max', window_name)
            
            # Crear rangos
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            # Crear máscara
            mask = cv2.inRange(hsv, lower, upper)
            
            # Aplicar morfología para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Crear visualización
            result = cv2.bitwise_and(image, image, mask=mask_clean)
            
            # Combinar vistas
            mask_colored = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            top_row = np.hstack([image, mask_colored])
            bottom_row = np.hstack([result, np.zeros_like(image)])
            
            combined = np.vstack([top_row, bottom_row])
            
            # Añadir texto con valores actuales
            text = f"HSV: [{h_min}-{h_max}, {s_min}-{s_max}, {v_min}-{v_max}]"
            cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Original | Mascara | Resultado | Info", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(combined, "Presiona 's' para GUARDAR | 'r' para RESET | 'q' para SALIR", 
                       (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2)
            
            # Mostrar
            cv2.imshow(window_name, combined)
            
            # Esperar tecla
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Guardar
                self.green_hsv_lower = lower
                self.green_hsv_upper = upper
                print("\n✅ Calibración GUARDADA:")
                print(f"   HSV Lower: {self.green_hsv_lower}")
                print(f"   HSV Upper: {self.green_hsv_upper}")
                
                # Guardar en archivo
                self._save_calibration({
                    'green_hsv_lower': self.green_hsv_lower.tolist(),
                    'green_hsv_upper': self.green_hsv_upper.tolist()
                })
                break
            
            elif key == ord('r'):  # Reset
                cv2.setTrackbarPos('H Min', window_name, 35)
                cv2.setTrackbarPos('H Max', window_name, 85)
                cv2.setTrackbarPos('S Min', window_name, 40)
                cv2.setTrackbarPos('S Max', window_name, 255)
                cv2.setTrackbarPos('V Min', window_name, 40)
                cv2.setTrackbarPos('V Max', window_name, 255)
                print("🔄 Valores reseteados a defaults")
            
            elif key == ord('q'):  # Salir sin guardar
                print("\n⚠️  Calibración cancelada (no guardada)")
                break
        
        cv2.destroyAllWindows()
        return True
    
    def _save_calibration(self, params: Dict):
        """Guarda parámetros de calibración en archivo JSON."""
        config_file = 'calibration_config.json'
        with open(config_file, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"💾 Calibración guardada en: {config_file}")
    
    def load_calibration(self, config_file: str = 'calibration_config.json') -> bool:
        """Carga parámetros de calibración desde archivo."""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                params = json.load(f)
            self.green_hsv_lower = np.array(params['green_hsv_lower'])
            self.green_hsv_upper = np.array(params['green_hsv_upper'])
            print(f"✅ Calibración cargada desde: {config_file}")
            print(f"   HSV Lower: {self.green_hsv_lower}")
            print(f"   HSV Upper: {self.green_hsv_upper}")
            return True
        return False
    
    def detect_cards(self, image: np.ndarray, debug: bool = False) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Detecta y segmenta todas las cartas en la imagen.
        """
        # Conversión a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Crear máscara del tapete verde
        green_mask = cv2.inRange(hsv, self.green_hsv_lower, self.green_hsv_upper)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (self.morph_kernel_size, self.morph_kernel_size))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Invertir máscara (cartas = todo lo que NO es verde)
        card_mask = cv2.bitwise_not(green_mask)
        
        # Detectar contornos
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        
        if debug:
            debug_img = image.copy()
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filtrar por área
            if area < self.min_card_area or area > self.max_card_area:
                continue
            
            # Rectángulo rotado
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Filtro de relación de aspecto
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if not (1.2 < aspect_ratio < 1.7):
                continue
            
            # Extraer carta
            card_img = self._extract_card(image, box, width, height)
            
            if card_img is not None:
                detected_cards.append((card_img, box))
                
                if debug:
                    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)
                    cv2.putText(debug_img, f"Card {idx+1}", 
                               tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 0, 0), 2)
        
        if debug:
            cv2.imwrite('detection_debug.png', debug_img)
            cv2.imwrite('card_mask_debug.png', card_mask)
            print(f"   🔍 Debug: detection_debug.png, card_mask_debug.png")
        
        print(f"🎴 Cartas detectadas: {len(detected_cards)}")
        return detected_cards
    
    def _extract_card(self, image: np.ndarray, box: np.ndarray, 
                     width: int, height: int) -> Optional[np.ndarray]:
        """Extrae carta con transformación perspectiva."""
        if width > height:
            width, height = height, width
        
        pts = box.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def recognize_card(self, card_image: np.ndarray, top_n: int = 3) -> List[Tuple[str, float]]:
        """Reconoce una carta comparándola con templates."""
        if len(self.templates) == 0:
            print("❌ No hay templates cargados")
            return []
        
        # Convertir a escala de grises
        if len(card_image.shape) == 3:
            card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            card_gray = card_image.copy()
        
        # Redimensionar
        card_resized = cv2.resize(card_gray, self.template_size, 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Normalizar
        card_normalized = cv2.normalize(card_resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Comparar con templates
        scores = {}
        
        for card_name, template in self.templates.items():
            # Correlación normalizada
            result = cv2.matchTemplate(card_normalized, template, cv2.TM_CCOEFF_NORMED)
            score_corr = result[0, 0]
            
            # Diferencia absoluta
            diff = cv2.absdiff(card_normalized, template)
            score_diff = 1.0 - (np.mean(diff) / 255.0)
            
            # Correlación estructural
            score_struct = np.corrcoef(card_normalized.flatten(), 
                                      template.flatten())[0, 1]
            
            # Score final combinado
            final_score = (0.5 * score_corr + 0.3 * score_diff + 0.2 * score_struct)
            scores[card_name] = final_score
        
        # Ordenar
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_n]
    
    def process_image(self, image_path: str, output_dir: str = "output", 
                     debug: bool = True) -> List[Dict]:
        """Procesa imagen completa."""
        print(f"\n{'='*70}")
        print(f"🎴 PROCESANDO: {image_path}")
        print(f"{'='*70}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error al cargar imagen")
            return []
        
        print(f"📐 Dimensiones: {image.shape[1]}x{image.shape[0]}")
        
        # Detectar
        detected_cards = self.detect_cards(image, debug=debug)
        
        if len(detected_cards) == 0:
            print("⚠️  No se detectaron cartas")
            return []
        
        # Reconocer
        results = []
        result_image = image.copy()
        
        for idx, (card_img, box) in enumerate(detected_cards):
            print(f"\n--- Carta {idx + 1} ---")
            
            # Guardar carta
            card_filename = os.path.join(output_dir, f"card_{idx+1}.png")
            cv2.imwrite(card_filename, card_img)
            
            # Reconocer
            matches = self.recognize_card(card_img, top_n=3)
            
            if len(matches) > 0:
                best_match, best_score = matches[0]
                
                print(f"🎯 Reconocida: {best_match} (score: {best_score:.3f})")
                print(f"   Top 3:")
                for rank, (name, score) in enumerate(matches, 1):
                    print(f"      {rank}. {name}: {score:.3f}")
                
                is_confident = best_score >= self.match_threshold
                color = (0, 255, 0) if is_confident else (0, 165, 255)
                
                cv2.drawContours(result_image, [box], 0, color, 3)
                label = f"{best_match}: {best_score:.2f}"
                cv2.putText(result_image, label, tuple(box[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                results.append({
                    'card_number': idx + 1,
                    'recognized_as': best_match,
                    'confidence': float(best_score),
                    'is_confident': is_confident,
                    'top_matches': [(name, float(score)) for name, score in matches]
                })
        
        # Guardar resultado
        result_path = os.path.join(output_dir, "result_annotated.png")
        cv2.imwrite(result_path, result_image)
        print(f"\n💾 Resultado: {result_path}")
        
        # Guardar JSON
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*70}")
        print(f"✅ {len(detected_cards)} cartas procesadas")
        print(f"   Confiables: {sum(1 for r in results if r['is_confident'])}")
        print(f"{'='*70}\n")
        
        return results


def main():
    """Función principal."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🎴  SISTEMA DE RECONOCIMIENTO DE CARTAS V2  🎴         ║
║                                                              ║
║        Con Calibración Interactiva del Tapete Verde         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Inicializar sistema
    system = CardRecognitionSystemV2(template_dir="templates")
    
    # Intentar cargar calibración previa
    if not system.load_calibration():
        print("\n⚠️  No hay calibración guardada.")
        print("   Usa system.calibrate_interactive('imagen_tapete.jpg')\n")
    
    return system


if __name__ == "__main__":
    system = main()