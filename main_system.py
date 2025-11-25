"""
SISTEMA PRINCIPAL DE DETECCIÓN DE CARTAS
Sistema completo e integrado - Listo para usar
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Set
from pathlib import Path

# Importar nuestros módulos
from template_matcher_pro import TemplateMatcherPro
from card_detector_pro import CardDetectorPro

class PokerCardSystem:
    """Sistema completo de detección de cartas de poker"""
    
    def __init__(self):
        print("\n" + "=" * 70)
        print("🃏 SISTEMA DE DETECCIÓN DE CARTAS DE POKER")
        print("=" * 70)
        
        # Inicializar componentes
        print("\n⚙️  Inicializando componentes...")
        
        self.detector = CardDetectorPro()
        print("✅ Detector de cartas: OK")
        
        self.matcher = TemplateMatcherPro()
        print("✅ Template matcher: OK")
        
        # Estado
        self.camera = None
        self.is_running = False
        self.detected_cards: Set[str] = set()
        self.last_process_time = 0
        self.process_interval = 0.5  # Procesar cada 0.5 seg
        
        # Verificar que todo esté listo
        self._verify_system()
    
    def _verify_system(self):
        """Verifica que el sistema esté completo"""
        print("\n🔍 Verificando sistema...")
        
        issues = []
        
        if not self.matcher.templates:
            issues.append("❌ No hay plantillas cargadas")
            issues.append("   → Verifica directorio 'templates/' con carpetas de palos")
        
        if issues:
            print("\n⚠️  PROBLEMAS DETECTADOS:")
            for issue in issues:
                print(issue)
            print("\n❌ Sistema NO puede funcionar correctamente")
            return False
        
        print("✅ Sistema verificado - TODO OK")
        print(f"   Plantillas: {len(self.matcher.templates)}")
        print("=" * 70)
        return True
    
    def calibrate_table(self, camera_index: int = 0):
        """Calibra el tapete verde"""
        print("\n🎨 Iniciando calibración del tapete...")
        return self.detector.calibrate(camera_index)
    
    def start_camera(self, camera_index: int = 0):
        """Inicia la cámara"""
        print(f"\n🚀 Iniciando cámara {camera_index}...")
        
        self.camera = cv2.VideoCapture(camera_index)
        
        if not self.camera.isOpened():
            print(f"❌ No se pudo abrir cámara {camera_index}")
            return False
        
        # Configurar
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Test
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("❌ Cámara no produce video")
            return False
        
        self.is_running = True
        print("✅ Cámara iniciada correctamente")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame: detecta y reconoce cartas"""
        
        # Detectar cartas
        card_images, positions = self.detector.detect_cards(frame)
        
        labels = []
        confidences = []
        
        # Reconocer cada carta
        for card_img in card_images:
            card_name, confidence, top_matches = self.matcher.recognize(card_img)
            
            if card_name and confidence >= 0.45:  # Umbral 45%
                labels.append(card_name)
                confidences.append(confidence)
                
                # Agregar a lista si alta confianza
                if confidence >= 0.60 and card_name not in self.detected_cards:
                    self.detected_cards.add(card_name)
                    print(f"\n🎴 DETECTADA: {card_name} ({confidence:.1%})")
                    
                    # Mostrar top 3
                    print("   Top 3:")
                    for i, match in enumerate(top_matches[:3], 1):
                        print(f"      {i}. {match['name']}: {match['score']:.3f}")
            else:
                labels.append("?")
                confidences.append(confidence if card_name else 0.0)
        
        # Dibujar
        result = self.detector.draw_detections(frame, positions, labels, confidences)
        
        # Info en pantalla
        cv2.putText(result, f"Cartas en mesa: {len(card_images)}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(result, f"Reconocidas: {len(self.detected_cards)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "q=Salir | r=Reset | s=Guardar", 
                   (10, result.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return result
    
    def run(self, camera_index: int = 0):
        """Ejecuta el sistema principal"""
        
        # Verificar plantillas
        if not self.matcher.templates:
            print("\n❌ ERROR: No hay plantillas")
            print("   Verifica que exista 'templates/' con las carpetas de palos")
            return
        
        # Iniciar cámara
        if not self.start_camera(camera_index):
            return
        
        print("\n" + "=" * 70)
        print("🎮 SISTEMA EN FUNCIONAMIENTO")
        print("=" * 70)
        print("\n📋 CONTROLES:")
        print("   q - Salir")
        print("   r - Resetear lista de cartas")
        print("   s - Guardar cartas detectadas")
        print("   c - Recalibrar tapete")
        print("=" * 70)
        print("\nColoca cartas sobre el tapete verde...")
        print("=" * 70)
        
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("\n⚠️  Error leyendo cámara")
                    break
                
                # Procesar cada cierto tiempo
                current_time = time.time()
                if current_time - self.last_process_time >= self.process_interval:
                    self.last_process_time = current_time
                    display_frame = self.process_frame(frame)
                else:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Cartas: {len(self.detected_cards)}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # FPS
                fps_counter += 1
                if current_time - fps_time >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = current_time
                
                cv2.putText(display_frame, f"FPS: {fps_display}", 
                           (display_frame.shape[1]-150, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Mostrar
                cv2.imshow("Sistema Deteccion Cartas Poker", display_frame)
                
                # Teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Saliendo...")
                    break
                
                elif key == ord('r'):
                    self.detected_cards.clear()
                    print("\n🧹 Lista limpiada")
                
                elif key == ord('s'):
                    self._save_results()
                
                elif key == ord('c'):
                    print("\n🎨 Recalibrando...")
                    self.camera.release()
                    cv2.destroyAllWindows()
                    self.detector.calibrate(camera_index)
                    if not self.start_camera(camera_index):
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por usuario")
        
        finally:
            self._cleanup()
    
    def _save_results(self):
        """Guarda cartas detectadas"""
        if not self.detected_cards:
            print("\n⚠️  No hay cartas para guardar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_poker_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🃏 CARTAS DE POKER DETECTADAS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total: {len(self.detected_cards)}\n\n")
                
                for i, card in enumerate(sorted(self.detected_cards), 1):
                    f.write(f"{i}. {card}\n")
            
            print(f"\n💾 Guardado: {filename}")
        
        except Exception as e:
            print(f"\n❌ Error guardando: {e}")
    
    def _cleanup(self):
        """Limpieza al cerrar"""
        print("\n🧹 Limpiando recursos...")
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE LA SESIÓN")
        print("=" * 70)
        
        if self.detected_cards:
            print(f"\n✅ Cartas reconocidas: {len(self.detected_cards)}")
            for i, card in enumerate(sorted(self.detected_cards), 1):
                print(f"   {i}. {card}")
        else:
            print("\n⚠️  No se reconocieron cartas")
        
        print("\n" + "=" * 70)
        print("👋 Sistema cerrado")
        print("=" * 70)


def main():
    """Función principal"""
    
    # Crear sistema
    system = PokerCardSystem()
    
    # Si no hay plantillas, no continuar
    if not system.matcher.templates:
        print("\n❌ SISTEMA NO PUEDE FUNCIONAR")
        print("\nAsegúrate de tener:")
        print("  templates/")
        print("    ├── corazones/")
        print("    ├── diamantes/")
        print("    ├── picas/")
        print("    └── treboles/")
        return
    
    # Menú
    print("\n" + "=" * 70)
    print("MENÚ PRINCIPAL")
    print("=" * 70)
    print("1. 🎮 Iniciar sistema")
    print("2. 🎨 Calibrar tapete verde")
    print("3. ❌ Salir")
    print("=" * 70)
    
    choice = input("\nOpción (1-3): ").strip()
    
    if choice == '1':
        cam = input("Índice de cámara (default=0): ").strip()
        cam_idx = int(cam) if cam else 0
        system.run(cam_idx)
    
    elif choice == '2':
        cam = input("Índice de cámara (default=0): ").strip()
        cam_idx = int(cam) if cam else 0
        if system.calibrate_table(cam_idx):
            print("\n✅ Calibración completada")
            ejecutar = input("\n¿Ejecutar sistema ahora? (s/n): ").lower()
            if ejecutar == 's':
                system.run(cam_idx)
    
    elif choice == '3':
        print("\n👋 ¡Hasta luego!")
    
    else:
        print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()