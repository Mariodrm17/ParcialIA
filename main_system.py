"""
SISTEMA PRINCIPAL OPTIMIZADO
Sin tirones + Mejor detección
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Set
from pathlib import Path
import threading

from template_matcher_pro import TemplateMatcherPro
from card_detector_pro import CardDetectorPro

class PokerCardSystem:
    """Sistema optimizado sin tirones"""
    
    def __init__(self):
        print("\n" + "=" * 70)
        print("🃏 SISTEMA DE DETECCIÓN DE CARTAS")
        print("=" * 70)
        
        self.detector = CardDetectorPro()
        print("✅ Detector: OK")
        
        self.matcher = TemplateMatcherPro()
        print("✅ Matcher: OK")
        
        # Estado
        self.camera = None
        self.is_running = False
        self.detected_cards: Set[str] = set()
        
        # Para procesamiento asíncrono (sin tirones)
        self.current_frame = None
        self.display_frame = None
        self.processing_lock = threading.Lock()
        self.process_thread = None
        
        self._verify_system()
    
    def _verify_system(self):
        """Verifica componentes"""
        if not self.matcher.templates:
            print("\n❌ NO HAY PLANTILLAS")
            return False
        
        print(f"✅ {len(self.matcher.templates)} plantillas cargadas")
        print("=" * 70)
        return True
    
    def start_camera(self, camera_index: int = 0):
        """Inicia cámara con configuración optimizada"""
        print(f"\n🚀 Iniciando cámara {camera_index}...")
        
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print("❌ No se pudo abrir cámara")
            return False
        
        # Configuración OPTIMIZADA para velocidad
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        
        # Test
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Cámara no produce video")
            return False
        
        self.is_running = True
        self.display_frame = frame
        
        # Iniciar thread de procesamiento
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        print("✅ Cámara iniciada")
        return True
    
    def _process_loop(self):
        """
        Loop de procesamiento en thread separado
        Esto evita los tirones
        """
        last_process = 0
        
        while self.is_running:
            current_time = time.time()
            
            # Procesar cada 1 segundo (ajustable)
            if current_time - last_process >= 1.0:
                with self.processing_lock:
                    if self.current_frame is not None:
                        processed = self._process_frame(self.current_frame.copy())
                        self.display_frame = processed
                
                last_process = current_time
            
            time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa frame: detecta y reconoce"""
        
        # Detectar cartas
        card_images, positions = self.detector.detect_cards(frame)
        
        labels = []
        confidences = []
        
        # Reconocer
        for card_img in card_images:
            card_name, confidence, top_matches = self.matcher.recognize(card_img)
            
            if card_name and confidence >= 0.40:  # Umbral 40%
                labels.append(card_name)
                confidences.append(confidence)
                
                # Agregar si alta confianza
                if confidence >= 0.55 and card_name not in self.detected_cards:
                    self.detected_cards.add(card_name)
                    print(f"\n🎴 {card_name} ({confidence:.1%})")
                    
                    # Top 3
                    if top_matches:
                        print("   Top 3:")
                        for i, m in enumerate(top_matches[:3], 1):
                            print(f"      {i}. {m['name']}: {m['score']:.3f}")
            else:
                labels.append("?")
                confidences.append(confidence if card_name else 0.0)
        
        # Dibujar
        result = self.detector.draw_detections(frame, positions, labels, confidences)
        
        # Info
        cv2.putText(result, f"En mesa: {len(card_images)}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(result, f"Reconocidas: {len(self.detected_cards)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result
    
    def run(self, camera_index: int = 0):
        """Ejecuta sistema"""
        
        if not self.matcher.templates:
            print("\n❌ No hay plantillas")
            return
        
        if not self.start_camera(camera_index):
            return
        
        print("\n" + "=" * 70)
        print("🎮 SISTEMA EN MARCHA")
        print("=" * 70)
        print("\n📋 CONTROLES:")
        print("   q - Salir")
        print("   r - Reset")
        print("   s - Guardar")
        print("   c - Recalibrar")
        print("=" * 70)
        
        fps_time = time.time()
        fps_count = 0
        fps_display = 0
        
        try:
            while self.is_running:
                # Leer frame (RÁPIDO, sin procesamiento pesado)
                ret, frame = self.camera.read()
                
                if not ret:
                    break
                
                # Guardar para procesamiento asíncrono
                with self.processing_lock:
                    self.current_frame = frame
                    show_frame = self.display_frame.copy() if self.display_frame is not None else frame
                
                # FPS
                fps_count += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps_display = fps_count
                    fps_count = 0
                    fps_time = current_time
                
                # Info adicional
                cv2.putText(show_frame, f"FPS: {fps_display}", 
                           (show_frame.shape[1]-150, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(show_frame, "q=Salir | r=Reset | s=Guardar | c=Calibrar", 
                           (10, show_frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Mostrar
                cv2.imshow("Sistema Deteccion Poker", show_frame)
                
                # Teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Saliendo...")
                    break
                
                elif key == ord('r'):
                    self.detected_cards.clear()
                    print("\n🧹 Reset")
                
                elif key == ord('s'):
                    self._save_results()
                
                elif key == ord('c'):
                    print("\n🎨 Recalibrando...")
                    self.is_running = False
                    self.camera.release()
                    cv2.destroyAllWindows()
                    time.sleep(0.5)
                    self.detector.calibrate(camera_index)
                    if self.start_camera(camera_index):
                        continue
                    else:
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido")
        
        finally:
            self._cleanup()
    
    def _save_results(self):
        """Guarda resultados"""
        if not self.detected_cards:
            print("\n⚠️  Nada que guardar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🃏 CARTAS DETECTADAS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total: {len(self.detected_cards)}\n\n")
                
                for i, card in enumerate(sorted(self.detected_cards), 1):
                    f.write(f"{i}. {card}\n")
            
            print(f"\n💾 Guardado: {filename}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def _cleanup(self):
        """Limpieza"""
        print("\n🧹 Cerrando...")
        
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2)
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("📊 RESUMEN")
        print("=" * 70)
        
        if self.detected_cards:
            print(f"\n✅ {len(self.detected_cards)} cartas:")
            for i, card in enumerate(sorted(self.detected_cards), 1):
                print(f"   {i}. {card}")
        else:
            print("\n⚠️  No se detectaron cartas")
        
        print("\n" + "=" * 70)
        print("👋 Cerrado")
        print("=" * 70)


def main():
    """Main"""
    system = PokerCardSystem()
    
    if not system.matcher.templates:
        print("\n❌ FALTA DIRECTORIO templates/")
        return
    
    print("\n" + "=" * 70)
    print("MENÚ")
    print("=" * 70)
    print("1. 🎮 Iniciar")
    print("2. 🎨 Calibrar tapete")
    print("3. ❌ Salir")
    print("=" * 70)
    
    choice = input("\nOpción: ").strip()
    
    if choice == '1':
        cam = input("Cámara (0): ").strip()
        cam_idx = int(cam) if cam else 0
        system.run(cam_idx)
    
    elif choice == '2':
        cam = input("Cámara (0): ").strip()
        cam_idx = int(cam) if cam else 0
        if system.detector.calibrate(cam_idx):
            print("\n✅ Calibrado")
            if input("\n¿Iniciar? (s/n): ").lower() == 's':
                system.run(cam_idx)
    
    elif choice == '3':
        print("\n👋 Adiós")


if __name__ == "__main__":
    main()