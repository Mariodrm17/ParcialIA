"""
CAMERA MANAGER - Gestión Robusta de Cámaras
Maneja detección, inicialización y captura de frames
"""

import cv2
import threading
import time
from typing import Optional, Callable, List, Dict

class CameraManager:
    """Gestor profesional de cámaras con thread-safety"""
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.available_cameras: List[Dict] = []
        self.frame_callback: Optional[Callable] = None
        self.lock = threading.Lock()
        self.capture_thread = None
        
    def scan_cameras(self, max_cameras: int = 4) -> List[Dict]:
        """
        Escanea y detecta cámaras disponibles
        
        Returns:
            Lista de diccionarios con info de cada cámara
        """
        self.available_cameras = []
        print("=" * 60)
        print("🔍 ESCANEANDO CÁMARAS")
        print("=" * 60)
        
        for i in range(max_cameras):
            try:
                # Probar abrir cámara
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    # Esperar inicialización
                    time.sleep(0.3)
                    
                    # Intentar leer un frame
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Obtener propiedades
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'index': i,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'name': f"Cámara {i} ({width}x{height})"
                        }
                        
                        self.available_cameras.append(camera_info)
                        print(f"✅ Cámara {i}: {width}x{height} @ {fps:.1f}FPS")
                    else:
                        print(f"⚠️  Cámara {i} abierta pero sin video")
                    
                    cap.release()
                else:
                    print(f"❌ Cámara {i} no disponible")
                    
            except Exception as e:
                # Ignorar errores silenciosamente
                pass
        
        print("=" * 60)
        print(f"✅ Total encontradas: {len(self.available_cameras)}")
        print("=" * 60)
        
        return self.available_cameras
    
    def start_camera(self, camera_index: int = 0, 
                    width: int = 1280, height: int = 720,
                    fps: int = 30) -> tuple[bool, str]:
        """
        Inicia la cámara con configuración específica
        
        Args:
            camera_index: Índice de la cámara
            width: Ancho deseado
            height: Alto deseado
            fps: FPS deseado
            
        Returns:
            (éxito, mensaje)
        """
        try:
            print(f"\n🚀 Iniciando cámara {camera_index}...")
            
            # Cerrar cámara anterior si existe
            if self.camera is not None:
                self.stop_camera()
                time.sleep(0.5)
            
            # Probar diferentes backends
            backends = [
                cv2.CAP_DSHOW,   # DirectShow (Windows)
                cv2.CAP_MSMF,    # Media Foundation (Windows)
                cv2.CAP_V4L2,    # Video4Linux (Linux)
                cv2.CAP_AVFOUNDATION,  # macOS
                cv2.CAP_ANY      # Auto-detect
            ]
            
            camera_opened = False
            
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(camera_index, backend)
                    
                    if self.camera.isOpened():
                        # Probar lectura
                        ret, test_frame = self.camera.read()
                        
                        if ret and test_frame is not None:
                            camera_opened = True
                            print(f"✅ Backend exitoso: {backend}")
                            break
                        else:
                            self.camera.release()
                            
                except Exception:
                    continue
            
            if not camera_opened:
                return False, f"No se pudo abrir la cámara {camera_index}"
            
            # Configurar propiedades
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Esperar estabilización
            time.sleep(1)
            
            # Verificar funcionamiento
            working_frames = 0
            for _ in range(5):
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    working_frames += 1
                time.sleep(0.1)
            
            if working_frames == 0:
                self.camera.release()
                return False, "Cámara abierta pero no produce video"
            
            # Iniciar captura en thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Obtener configuración real
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            msg = f"Cámara {camera_index} iniciada: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS"
            print(f"✅ {msg}")
            
            return True, msg
            
        except Exception as e:
            error_msg = f"Error crítico: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.camera:
                self.camera.release()
                self.camera = None
                
            return False, error_msg
    
    def _capture_loop(self):
        """Loop de captura en thread separado"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.is_running and self.camera:
            try:
                with self.lock:
                    ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    self.current_frame = frame
                    consecutive_errors = 0
                    
                    # Llamar callback si existe
                    if self.frame_callback:
                        try:
                            self.frame_callback(frame)
                        except Exception as e:
                            print(f"⚠️  Error en callback: {e}")
                else:
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_errors:
                        print(f"❌ Demasiados errores ({max_errors}), deteniendo")
                        self.stop_camera()
                        break
                        
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error en captura: {e}")
                
                if consecutive_errors >= max_errors:
                    self.stop_camera()
                    break
                    
                time.sleep(0.1)
    
    def stop_camera(self):
        """Detiene la cámara de forma segura"""
        print("🛑 Deteniendo cámara...")
        
        self.is_running = False
        
        # Esperar thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Liberar cámara
        if self.camera:
            with self.lock:
                self.camera.release()
            self.camera = None
        
        self.current_frame = None
        print("✅ Cámara detenida")
    
    def get_frame(self) -> Optional[any]:
        """Obtiene el frame actual de forma segura"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def set_frame_callback(self, callback: Callable):
        """Establece callback para procesar cada frame"""
        self.frame_callback = callback
    
    def is_camera_active(self) -> bool:
        """Verifica si la cámara está activa y funcionando"""
        return (self.is_running and 
                self.camera is not None and 
                self.current_frame is not None)
    
    def get_camera_info(self) -> Optional[Dict]:
        """Obtiene información actual de la cámara"""
        if not self.camera:
            return None
        
        try:
            return {
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'is_running': self.is_running
            }
        except:
            return None


# Test del módulo
if __name__ == "__main__":
    print("🧪 TEST: Camera Manager")
    
    manager = CameraManager()
    
    # Escanear cámaras
    cameras = manager.scan_cameras()
    
    if cameras:
        # Usar primera cámara
        cam_idx = cameras[0]['index']
        
        success, msg = manager.start_camera(cam_idx)
        
        if success:
            print(f"\n✅ {msg}")
            print("Mostrando video por 5 segundos...")
            
            start = time.time()
            while time.time() - start < 5:
                frame = manager.get_frame()
                
                if frame is not None:
                    cv2.imshow("Test Camera Manager", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            manager.stop_camera()
            cv2.destroyAllWindows()
            print("✅ Test completado")
        else:
            print(f"❌ {msg}")
    else:
        print("❌ No hay cámaras disponibles")