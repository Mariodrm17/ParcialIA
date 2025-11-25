"""
GREEN CALIBRATOR - VERSIÓN ARREGLADA
Usa múltiples backends como CameraManager
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

class GreenCalibrator:
    """Calibrador interactivo del tapete verde"""
    
    def __init__(self, config_file: str = "green_calibration.json"):
        self.config_file = Path(config_file)
        self.green_lower = None
        self.green_upper = None
    
    def _open_camera(self, camera_index: int):
        """Abre cámara probando múltiples backends"""
        backends = [
            cv2.CAP_DSHOW,   # DirectShow (Windows)
            cv2.CAP_MSMF,    # Media Foundation (Windows)
            cv2.CAP_V4L2,    # Video4Linux (Linux)
            cv2.CAP_AVFOUNDATION,  # macOS
            cv2.CAP_ANY      # Auto-detect
        ]
        
        for backend in backends:
            try:
                print(f"  Probando backend: {backend}")
                cap = cv2.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    # Verificar que funciona
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        print(f"  ✅ Backend exitoso: {backend}")
                        return cap
                    else:
                        cap.release()
                        
            except Exception as e:
                print(f"  ❌ Falló backend {backend}: {e}")
                continue
        
        return None
        
    def calibrate(self, camera_index: int = 0) -> bool:
        """
        Calibración interactiva con trackbars
        """
        print("=" * 70)
        print("🎨 CALIBRADOR DE TAPETE VERDE")
        print("=" * 70)
        print("\n📋 INSTRUCCIONES:")
        print("1. Coloca SOLO el tapete verde frente a la cámara")
        print("2. Asegúrate de buena iluminación uniforme")
        print("3. NO debe haber cartas ni objetos sobre el tapete")
        print("4. Ajusta los trackbars hasta que:")
        print("   ✅ El tapete verde aparezca COMPLETAMENTE BLANCO")
        print("   ✅ Todo lo demás aparezca NEGRO")
        print("5. Presiona 's' para GUARDAR")
        print("6. Presiona 'q' para CANCELAR")
        print("=" * 70)
        
        input("\n⏸️  Presiona ENTER cuando esté listo...")
        
        # Abrir cámara CON MÚLTIPLES BACKENDS
        print(f"\n🎥 Abriendo cámara {camera_index}...")
        cap = self._open_camera(camera_index)
        
        if cap is None:
            print(f"\n❌ NO SE PUDO ABRIR LA CÁMARA {camera_index}")
            print("\n🔧 SOLUCIONES:")
            print("1. Verifica que la cámara esté conectada")
            print("2. Cierra otras aplicaciones que usen la cámara")
            print("3. Prueba con otro índice (python 2_green_calibrator.py)")
            print("4. En Windows: verifica permisos de cámara en Configuración")
            return False
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Esperar inicialización
        time.sleep(1)
        
        print("✅ Cámara iniciada")
        
        # Valores iniciales (rango amplio de verdes)
        h_min, h_max = 35, 85
        s_min, s_max = 40, 255
        v_min, v_max = 40, 255
        
        # Crear ventanas
        window_original = "1. Original - Tu tapete"
        window_controls = "2. Controles - Ajusta aqui"
        window_result = "3. Resultado - BLANCO = tapete detectado"
        
        cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_controls, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
        
        # Crear trackbars
        cv2.createTrackbar('H Min', window_controls, h_min, 180, lambda x: None)
        cv2.createTrackbar('H Max', window_controls, h_max, 180, lambda x: None)
        cv2.createTrackbar('S Min', window_controls, s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', window_controls, s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', window_controls, v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', window_controls, v_max, 255, lambda x: None)
        
        # Crear imagen de ayuda
        control_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(control_img, "AJUSTA LOS TRACKBARS:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(control_img, "H = Tono del color (verde ~35-85)", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(control_img, "S = Saturacion (intensidad)", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(control_img, "V = Valor (brillo)", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(control_img, "Presiona 's' = GUARDAR | 'q' = CANCELAR", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        print("\n🔧 Ajustando parámetros...")
        print("💡 TIPS:")
        print("   - Empieza con H (Hue) para el tono de verde")
        print("   - Ajusta S (Saturation) para la intensidad del color")
        print("   - Finalmente V (Value) para el brillo")
        
        calibrated = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"⚠️  Frame {frame_count} falló")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Leer valores actuales
            h_min = cv2.getTrackbarPos('H Min', window_controls)
            h_max = cv2.getTrackbarPos('H Max', window_controls)
            s_min = cv2.getTrackbarPos('S Min', window_controls)
            s_max = cv2.getTrackbarPos('S Max', window_controls)
            v_min = cv2.getTrackbarPos('V Min', window_controls)
            v_max = cv2.getTrackbarPos('V Max', window_controls)
            
            # Crear rangos
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morfología
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
            
            # Calcular porcentaje
            white_pixels = cv2.countNonZero(mask_clean)
            total_pixels = mask_clean.shape[0] * mask_clean.shape[1]
            percentage = (white_pixels / total_pixels) * 100
            
            # Frame original con info
            info_frame = frame.copy()
            cv2.putText(info_frame, f"Tapete detectado: {percentage:.1f}%",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(info_frame, "'s' = Guardar | 'q' = Cancelar",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resultado con info
            result_display = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            cv2.putText(result_display, f"H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Estado visual
            if 20 <= percentage <= 80:
                color = (0, 255, 0)
                status = "BUENO"
            elif percentage < 20:
                color = (0, 165, 255)
                status = "MUY POCO - Aumenta rangos"
            else:
                color = (0, 0, 255)
                status = "DEMASIADO - Reduce rangos"
            
            cv2.putText(result_display, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Mostrar
            cv2.imshow(window_original, info_frame)
            cv2.imshow(window_controls, control_img)
            cv2.imshow(window_result, result_display)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n❌ Calibración cancelada")
                break
                
            elif key == ord('s'):
                # Validar
                if percentage < 10:
                    print(f"\n⚠️  Solo {percentage:.1f}% detectado. Demasiado poco.")
                    continue
                
                if percentage > 95:
                    print(f"\n⚠️  {percentage:.1f}% detectado. Demasiado.")
                    continue
                
                # Guardar
                self.green_lower = lower
                self.green_upper = upper
                
                print("\n✅ CALIBRACIÓN EXITOSA!")
                print(f"📊 Tapete detectado: {percentage:.1f}%")
                print(f"📝 Valores:")
                print(f"   H: [{h_min} - {h_max}]")
                print(f"   S: [{s_min} - {s_max}]")
                print(f"   V: [{v_min} - {v_max}]")
                
                self.save()
                calibrated = True
                break
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        return calibrated
    
    def save(self):
        """Guarda calibración"""
        if self.green_lower is None or self.green_upper is None:
            return False
        
        config = {
            'green_lower': self.green_lower.tolist(),
            'green_upper': self.green_upper.tolist(),
            'timestamp': datetime.now().isoformat(),
            'date_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"💾 Calibración guardada: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
            return False
    
    def load(self) -> bool:
        """Carga calibración"""
        if not self.config_file.exists():
            print(f"⚠️  No existe: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.green_lower = np.array(config['green_lower'])
            self.green_upper = np.array(config['green_upper'])
            
            print("✅ Calibración cargada")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando: {e}")
            return False
    
    def test_calibration(self, camera_index: int = 0):
        """Prueba calibración"""
        if self.green_lower is None or self.green_upper is None:
            print("❌ Primero carga o calibra")
            return
        
        print("\n🧪 PROBANDO CALIBRACIÓN")
        print("Presiona 'q' para salir")
        
        cap = self._open_camera(camera_index)
        
        if cap is None:
            print(f"❌ No se pudo abrir cámara {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Limpiar
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir
            cards_mask = cv2.bitwise_not(mask)
            
            # Mostrar
            cv2.imshow("Original", frame)
            cv2.imshow("Tapete (blanco = detectado)", mask)
            cv2.imshow("Cartas (blanco = posibles cartas)", cards_mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test finalizado")


def main():
    """Menú principal"""
    calibrator = GreenCalibrator()
    
    while True:
        print("\n" + "=" * 60)
        print("🎯 CALIBRADOR DE TAPETE VERDE")
        print("=" * 60)
        print("1. ✨ Calibrar nuevo tapete")
        print("2. 📂 Cargar calibración existente")
        print("3. 🧪 Probar calibración")
        print("4. ❌ Salir")
        print("=" * 60)
        
        choice = input("\nOpción (1-4): ").strip()
        
        if choice == '1':
            cam = input("Índice de cámara (default=0): ").strip()
            cam_idx = int(cam) if cam else 0
            
            success = calibrator.calibrate(cam_idx)
            
            if success:
                test = input("\n¿Probar calibración? (s/n): ").lower()
                if test == 's':
                    calibrator.test_calibration(cam_idx)
        
        elif choice == '2':
            success = calibrator.load()
            
            if success:
                test = input("\n¿Probar calibración? (s/n): ").lower()
                if test == 's':
                    cam = input("Índice de cámara (default=0): ").strip()
                    cam_idx = int(cam) if cam else 0
                    calibrator.test_calibration(cam_idx)
        
        elif choice == '3':
            if calibrator.green_lower is None:
                print("\n⚠️  Primero calibra o carga una calibración")
            else:
                cam = input("Índice de cámara (default=0): ").strip()
                cam_idx = int(cam) if cam else 0
                calibrator.test_calibration(cam_idx)
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()
