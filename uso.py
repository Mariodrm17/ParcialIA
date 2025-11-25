#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Uso - Sistema de Reconocimiento
"""

import cv2
from card_recognition_system import CardRecognitionSystem


def main():
    """Uso directo del sistema."""
    
    # Crear sistema
    system = CardRecognitionSystem(template_dir="templates")
    
    print("\n¿Qué quieres hacer?")
    print("1. Calibrar tapete verde")
    print("2. Procesar imagen con cartas")
    print("3. Ambas cosas")
    
    opcion = input("\nOpción (1/2/3): ").strip()
    
    if opcion in ['1', '3']:
        # CALIBRACIÓN
        print("\n--- CALIBRACIÓN ---")
        print("Opciones:")
        print("a) Capturar foto del tapete ahora")
        print("b) Usar imagen existente")
        
        cal_op = input("Opción (a/b): ").strip().lower()
        
        if cal_op == 'a':
            print("\nCapturando en 3...")
            input("Presiona ENTER cuando estés listo...")
            
            cap = cv2.VideoCapture(1)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('tapete.jpg', frame)
                print("✅ Guardado: tapete.jpg")
                cap.release()
                system.calibrate_interactive('tapete.jpg')
            else:
                print("❌ Error con cámara")
                cap.release()
        
        elif cal_op == 'b':
            img = input("\nRuta de imagen: ").strip()
            system.calibrate_interactive(img)
    
    if opcion in ['2', '3']:
        # PROCESAR
        print("\n--- PROCESAMIENTO ---")
        
        # Cargar calibración si existe
        system.load_calibration()
        
        print("Opciones:")
        print("a) Capturar foto de cartas ahora")
        print("b) Usar imagen existente")
        
        proc_op = input("Opción (a/b): ").strip().lower()
        
        if proc_op == 'a':
            print("\nCapturando en 3...")
            input("Presiona ENTER cuando estés listo...")
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('cartas.jpg', frame)
                print("✅ Guardado: cartas.jpg")
                cap.release()
                system.process_image('cartas.jpg')
            else:
                print("❌ Error con cámara")
                cap.release()
        
        elif proc_op == 'b':
            img = input("\nRuta de imagen: ").strip()
            system.process_image(img)
    
    print("\n✅ TERMINADO\n")


if __name__ == "__main__":
    main()