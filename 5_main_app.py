"""
MAIN APP - Aplicación Principal con GUI
Sistema completo de detección de cartas con interfaz gráfica
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from datetime import datetime
from typing import Set

# Importar nuestros módulos
from camera_manager import CameraManager
from card_detector import CardDetector
from template_matcher import TemplateMatcher

class CardDetectionApp:
    """Aplicación principal con GUI Tkinter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🃏 Sistema de Detección de Cartas Profesional")
        self.root.geometry("1400x900")
        
        # Componentes del sistema
        self.camera_manager = CameraManager()
        self.card_detector = CardDetector()
        self.template_matcher = TemplateMatcher()
        
        # Estado
        self.is_running = False
        self.current_frame = None
        self.detected_cards: Set[str] = set()
        self.last_process_time = 0
        self.process_interval = 1.0  # Procesar cada 1 segundo
        self.debug_mode = False
        
        # Crear interfaz
        self.create_gui()
        
        # Escanear cámaras al inicio
        self.scan_cameras()
        
        # Iniciar loop de actualización
        self.update_display()
        
        # Manejo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_gui(self):
        """Crea la interfaz gráfica"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ====== PANEL DE CONTROL ======
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Panel de Control", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Fila 1: Cámaras
        cam_frame = ttk.Frame(control_frame)
        cam_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(cam_frame, text="Cámara:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_frame, textvariable=self.cam_var,
                                     state="readonly", width=40)
        self.cam_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(cam_frame, text="🔍 Escanear", 
                  command=self.scan_cameras).pack(side=tk.LEFT, padx=(0, 5))
        
        # Fila 2: Controles principales
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_frame, text="▶️ Iniciar", 
                                   command=self.start_camera, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Detener", 
                                  command=self.stop_camera, state="disabled", width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="🧹 Limpiar Lista", 
                  command=self.clear_detections, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="💾 Guardar", 
                  command=self.save_results, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        self.debug_var = tk.BooleanVar()
        ttk.Checkbutton(btn_frame, text="🔧 Debug", 
                       variable=self.debug_var,
                       command=self.toggle_debug).pack(side=tk.LEFT, padx=(10, 0))
        
        # ====== CONTENIDO PRINCIPAL ======
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo: Cámara
        left_panel = ttk.LabelFrame(content_frame, text="📹 Vista en Vivo", padding="5")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Label para video
        self.video_label = ttk.Label(left_panel, text="Sistema iniciado\n\n"
                                     "1. Selecciona cámara\n"
                                     "2. Haz clic en Iniciar\n"
                                     "3. Coloca cartas sobre tapete verde\n\n"
                                     "⚙️  Estado del sistema:\n"
                                     f"{'✅' if self.card_detector.is_calibrated else '❌'} Calibración tapete\n"
                                     f"{'✅' if self.template_matcher.templates else '❌'} Plantillas ({len(self.template_matcher.templates)})",
                                     background="black", foreground="white",
                                     justify=tk.CENTER, font=("Arial", 11))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho: Resultados
        right_panel = ttk.Frame(content_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)
        
        # Cartas detectadas
        cards_frame = ttk.LabelFrame(right_panel, text="🎴 Cartas Detectadas", padding="5")
        cards_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Lista con scrollbar
        list_frame = ttk.Frame(cards_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.cards_listbox = tk.Listbox(list_frame, font=("Arial", 11),
                                        yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.cards_listbox.yview)
        
        self.cards_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Estadísticas
        stats_frame = ttk.LabelFrame(right_panel, text="📊 Estadísticas", padding="10")
        stats_frame.pack(fill=tk.X)
        
        self.stats_text = tk.Text(stats_frame, height=8, font=("Courier", 10),
                                 state='disabled', bg='#f0f0f0')
        self.stats_text.pack(fill=tk.X)
        
        # ====== BARRA DE ESTADO ======
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Sistema listo")
        ttk.Label(status_frame, textvariable=self.status_var,
                 relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var,
                 relief=tk.SUNKEN, width=15).pack(side=tk.RIGHT)
    
    def scan_cameras(self):
        """Escanea cámaras disponibles"""
        self.status_var.set("🔍 Escaneando cámaras...")
        self.root.update()
        
        cameras = self.camera_manager.scan_cameras()
        
        if cameras:
            cam_list = [f"{cam['name']}" for cam in cameras]
            self.cam_combo['values'] = cam_list
            self.cam_combo.current(0)
            self.status_var.set(f"✅ {len(cameras)} cámara(s) encontrada(s)")
        else:
            self.cam_combo['values'] = []
            self.status_var.set("❌ No se encontraron cámaras")
            messagebox.showwarning("Advertencia", "No se encontraron cámaras disponibles")
    
    def start_camera(self):
        """Inicia la cámara"""
        # Verificar calibración
        if not self.card_detector.is_calibrated:
            messagebox.showerror("Error", 
                               "⚠️  No hay calibración del tapete verde\n\n"
                               "Ejecuta primero:\n"
                               "python 2_green_calibrator.py")
            return
        
        # Verificar plantillas
        if not self.template_matcher.templates:
            messagebox.showerror("Error",
                               "⚠️  No hay plantillas cargadas\n\n"
                               "Verifica que exista el directorio 'templates'\n"
                               "con las carpetas: corazones, diamantes, picas, treboles")
            return
        
        # Verificar selección de cámara
        if not self.cam_combo.get():
            messagebox.showwarning("Advertencia", "Selecciona una cámara")
            return
        
        try:
            # Extraer índice de cámara
            selected = self.cam_var.get()
            # El nombre es "Cámara X (resolución)"
            cam_index = int(selected.split()[1])
            
            # Iniciar cámara
            self.status_var.set(f"🚀 Iniciando cámara {cam_index}...")
            self.root.update()
            
            success, message = self.camera_manager.start_camera(cam_index)
            
            if success:
                self.is_running = True
                self.status_var.set(f"✅ {message}")
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
                
                print("\n" + "=" * 60)
                print("🎯 SISTEMA INICIADO")
                print("=" * 60)
                print("Coloca cartas sobre el tapete verde...")
                print("=" * 60)
            else:
                self.status_var.set(f"❌ {message}")
                messagebox.showerror("Error", f"No se pudo iniciar cámara:\n{message}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar cámara:\n{str(e)}")
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.is_running = False
        self.camera_manager.stop_camera()
        
        self.video_label.config(image='')
        self.video_label.configure(text="⏹️ Cámara detenida\n\nHaz clic en 'Iniciar' para continuar")
        
        self.status_var.set("⏹️ Cámara detenida")
        self.fps_var.set("FPS: 0")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def toggle_debug(self):
        """Activa/desactiva modo debug"""
        self.debug_mode = self.debug_var.get()
        if self.debug_mode:
            print("\n🔧 Modo DEBUG activado - se mostrarán detalles de reconocimiento")
        else:
            print("\n🔧 Modo DEBUG desactivado")
    
    def process_frame(self, frame):
        """Procesa frame para detectar y reconocer cartas"""
        try:
            # Detectar cartas
            card_images, card_positions = self.card_detector.detect_cards(frame)
            
            labels = []
            confidences = []
            
            # Reconocer cada carta
            for card_img in card_images:
                card_name, confidence, top_matches = self.template_matcher.recognize_card(card_img)
                
                if card_name and confidence >= 0.50:  # Umbral mínimo 50%
                    labels.append(card_name)
                    confidences.append(confidence)
                    
                    # Agregar a lista si confianza alta
                    if confidence >= 0.65 and card_name not in self.detected_cards:
                        self.detected_cards.add(card_name)
                        print(f"🎴 Nueva carta: {card_name} ({confidence:.1%})")
                        
                        # Modo debug: mostrar top 3
                        if self.debug_mode:
                            print("   Top 3 coincidencias:")
                            for i, match in enumerate(top_matches, 1):
                                print(f"      {i}. {match['name']}: {match['score']:.4f}")
                else:
                    labels.append("?")
                    confidences.append(confidence if card_name else 0.0)
            
            # Dibujar detecciones
            display_frame = self.card_detector.draw_detections(
                frame, card_positions, labels, confidences
            )
            
            # Info en pantalla
            cv2.putText(display_frame, f"Cartas visibles: {len(card_images)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Reconocidas: {len(self.detected_cards)}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return display_frame
            
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            return frame
    
    def update_display(self):
        """Actualiza la visualización (loop principal)"""
        try:
            if self.is_running:
                frame = self.camera_manager.get_frame()
                
                if frame is not None:
                    # Procesar cada cierto tiempo
                    current_time = time.time()
                    if current_time - self.last_process_time >= self.process_interval:
                        self.last_process_time = current_time
                        display_frame = self.process_frame(frame)
                    else:
                        # Solo dibujar detecciones previas
                        display_frame = frame.copy()
                    
                    # Redimensionar para GUI
                    display_frame = cv2.resize(display_frame, (960, 720))
                    
                    # Convertir BGR a RGB
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convertir a ImageTk
                    img = Image.fromarray(rgb_frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    
                    # Actualizar FPS
                    cam_info = self.camera_manager.get_camera_info()
                    if cam_info:
                        self.fps_var.set(f"FPS: {cam_info['fps']:.1f}")
            
            # Actualizar lista de cartas
            self.update_cards_list()
            
            # Actualizar estadísticas
            self.update_stats()
            
            # Siguiente actualización
            self.root.after(30, self.update_display)
            
        except Exception as e:
            print(f"❌ Error en update_display: {e}")
            self.root.after(100, self.update_display)
    
    def update_cards_list(self):
        """Actualiza la lista de cartas detectadas"""
        self.cards_listbox.delete(0, tk.END)
        
        for card in sorted(self.detected_cards):
            self.cards_listbox.insert(tk.END, f"  • {card}")
    
    def update_stats(self):
        """Actualiza panel de estadísticas"""
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        
        stats = f"""
Sistema:      {'🟢 Activo' if self.is_running else '🔴 Detenido'}
Cartas:       {len(self.detected_cards)}
Plantillas:   {len(self.template_matcher.templates)}
Calibración:  {'✅ OK' if self.card_detector.is_calibrated else '❌ No'}
Debug:        {'🔧 ON' if self.debug_mode else 'OFF'}
        """
        
        self.stats_text.insert('1.0', stats)
        self.stats_text.config(state='disabled')
    
    def clear_detections(self):
        """Limpia lista de cartas detectadas"""
        self.detected_cards.clear()
        self.status_var.set("🧹 Lista limpiada")
        print("\n🧹 Lista de cartas limpiada")
    
    def save_results(self):
        """Guarda cartas detectadas en archivo"""
        if not self.detected_cards:
            messagebox.showinfo("Información", "No hay cartas para guardar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_detectadas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🃏 CARTAS DETECTADAS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total: {len(self.detected_cards)}\n\n")
                
                for i, card in enumerate(sorted(self.detected_cards), 1):
                    f.write(f"{i}. {card}\n")
            
            self.status_var.set(f"💾 Guardado: {filename}")
            messagebox.showinfo("Éxito", f"Cartas guardadas en:\n{filename}")
            print(f"\n💾 Cartas guardadas: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar:\n{str(e)}")
    
    def on_closing(self):
        """Manejo de cierre de la aplicación"""
        if self.is_running:
            self.camera_manager.stop_camera()
        
        print("\n👋 Sistema cerrado")
        self.root.destroy()


def main():
    """Función principal"""
    print("=" * 70)
    print("🃏 SISTEMA DE DETECCIÓN DE CARTAS - VERSIÓN PROFESIONAL")
    print("=" * 70)
    print("\n⚙️  Verificando componentes...")
    
    # Verificar archivos necesarios
    from pathlib import Path
    
    issues = []
    
    if not Path("green_calibration.json").exists():
        issues.append("❌ No hay calibración del tapete (ejecuta: python 2_green_calibrator.py)")
    else:
        print("✅ Calibración del tapete: OK")
    
    if not Path("templates").exists():
        issues.append("❌ No existe directorio 'templates'")
    else:
        print("✅ Directorio de plantillas: OK")
    
    if issues:
        print("\n⚠️  ADVERTENCIAS:")
        for issue in issues:
            print(f"   {issue}")
        print("\nEl sistema puede no funcionar correctamente.")
        print("=" * 70)
    
    print("\n🚀 Iniciando aplicación...")
    print("=" * 70)
    
    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = CardDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()