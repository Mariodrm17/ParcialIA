#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Prueba y Utilidades
Sistema de Reconocimiento de Cartas
"""

import cv2
import numpy as np
from card_recognition_system import CardRecognitionSystem
import os


def test_calibration():
    """Test de calibración del tapete verde."""
    print("\n🎯 TEST: Calibración de Tapete Verde\n")
    
    # Crear imagen de prueba con tapete verde
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Verde del tapete (típico)
    test_img[:, :] = (60, 180, 80)  # BGR
    
    # Añadir una "carta" blanca en el centro
    cv2.rectangle(test_img, (300, 150), (500, 450), (255, 255, 255), -1)
    cv2.rectangle(test_img, (300, 150), (500, 450), (0, 0, 0), 3)
    
    cv2.imwrite('test_calibration_image.jpg', test_img)
    print("✅ Imagen de prueba creada: test_calibration_image.jpg")
    
    # Calibrar
    system = CardRecognitionSystem()
    params = system.calibrate_green_detection(test_img, show_results=True)
    
    print("\n✅ Calibración completada")
    return system


def test_single_card():
    """Test de reconocimiento de una sola carta."""
    print("\n🎴 TEST: Reconocimiento de Carta Individual\n")
    
    system = CardRecognitionSystem()
    
    # Verificar si existen templates
    if len(system.templates) == 0:
        print("❌ No hay templates cargados.")
        print("   Por favor, coloca las imágenes de cartas en la carpeta 'templates/'")
        return
    
    # Crear imagen de prueba
    test_img = create_test_scene(num_cards=1)
    cv2.imwrite('test_single_card.jpg', test_img)
    
    # Procesar
    results = system.process_image('test_single_card.jpg', output_dir='output_single')
    
    if results:
        print(f"\n✅ Test completado. Carta reconocida: {results[0]['recognized_as']}")
    
    return results


def test_multiple_cards():
    """Test de reconocimiento de múltiples cartas."""
    print("\n🎴🎴🎴 TEST: Reconocimiento de Múltiples Cartas\n")
    
    system = CardRecognitionSystem()
    
    if len(system.templates) == 0:
        print("❌ No hay templates cargados.")
        return
    
    # Crear imagen con 3 cartas
    test_img = create_test_scene(num_cards=3)
    cv2.imwrite('test_multiple_cards.jpg', test_img)
    
    # Procesar
    results = system.process_image('test_multiple_cards.jpg', output_dir='output_multiple')
    
    if results:
        print(f"\n✅ Test completado. {len(results)} cartas reconocidas")
    
    return results


def create_test_scene(num_cards=1, tapete_color=(60, 180, 80)):
    """
    Crea una escena de prueba con tapete verde y cartas blancas.
    
    Args:
        num_cards: Número de cartas a colocar
        tapete_color: Color BGR del tapete
        
    Returns:
        Imagen de la escena
    """
    # Crear tapete verde
    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    img[:, :] = tapete_color
    
    # Dimensiones de carta
    card_w, card_h = 226, 314
    
    # Posiciones para las cartas
    if num_cards == 1:
        positions = [(500, 250)]
    elif num_cards == 2:
        positions = [(300, 250), (700, 250)]
    elif num_cards == 3:
        positions = [(250, 250), (500, 250), (750, 250)]
    else:
        # Distribuir en grid
        cols = int(np.ceil(np.sqrt(num_cards)))
        rows = int(np.ceil(num_cards / cols))
        positions = []
        for i in range(num_cards):
            row = i // cols
            col = i % cols
            x = 200 + col * 300
            y = 150 + row * 350
            positions.append((x, y))
    
    # Dibujar cartas
    for idx, (x, y) in enumerate(positions):
        # Fondo blanco de carta
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (255, 255, 255), -1)
        
        # Borde negro
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (0, 0, 0), 3)
        
        # Texto simulando carta
        cv2.putText(img, f"CARTA", (x + 50, y + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, f"#{idx+1}", (x + 70, y + 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
    
    return img


def create_template_examples():
    """
    Crea ejemplos de templates para testing.
    """
    print("\n🎨 Creando templates de ejemplo...\n")
    
    os.makedirs('templates', exist_ok=True)
    
    # Cartas de ejemplo
    example_cards = ['AS', '2C', 'KH', 'QD', '10S', 'JC']
    
    for card_name in example_cards:
        # Crear imagen de carta
        card = np.ones((314, 226, 3), dtype=np.uint8) * 255  # Fondo blanco
        
        # Borde
        cv2.rectangle(card, (5, 5), (220, 308), (0, 0, 0), 3)
        
        # Valor de la carta
        value = ''.join([c for c in card_name if not c.isalpha()])
        if not value:
            value = card_name[0]  # Para A, K, Q, J
        
        suit = card_name[-1]  # Último carácter es el palo
        
        # Símbolos de palo
        suit_symbols = {
            'S': '♠',  # Picas (Spades)
            'H': '♥',  # Corazones (Hearts)
            'D': '♦',  # Diamantes (Diamonds)
            'C': '♣'   # Tréboles (Clubs)
        }
        
        suit_colors = {
            'S': (0, 0, 0),      # Negro
            'H': (0, 0, 255),    # Rojo
            'D': (0, 0, 255),    # Rojo
            'C': (0, 0, 0)       # Negro
        }
        
        suit_symbol = suit_symbols.get(suit, suit)
        suit_color = suit_colors.get(suit, (0, 0, 0))
        
        # Esquina superior izquierda
        cv2.putText(card, value, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, suit_color, 3)
        cv2.putText(card, suit_symbol, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, suit_color, 2)
        
        # Centro
        cv2.putText(card, suit_symbol, (80, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, suit_color, 4)
        
        # Esquina inferior derecha (invertida)
        cv2.putText(card, value, (180, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, suit_color, 3)
        cv2.putText(card, suit_symbol, (180, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, suit_color, 2)
        
        # Guardar
        filename = f'templates/{card_name}.png'
        cv2.imwrite(filename, card)
        print(f"✅ Creado: {filename}")
    
    print(f"\n✅ {len(example_cards)} templates de ejemplo creados en 'templates/'")


def show_template_info():
    """Muestra información sobre los templates cargados."""
    print("\n📚 Información de Templates\n")
    
    system = CardRecognitionSystem()
    
    if len(system.templates) == 0:
        print("❌ No hay templates cargados")
        print("   Ejecuta create_template_examples() para crear ejemplos")
        return
    
    print(f"Total de templates: {len(system.templates)}")
    print(f"Tamaño estándar: {system.template_size}")
    print("\nCartas disponibles:")
    
    for idx, card_name in enumerate(sorted(system.card_names), 1):
        print(f"  {idx:2d}. {card_name}")


def interactive_calibration():
    """Calibración interactiva con imagen de usuario."""
    print("\n🎯 Calibración Interactiva del Tapete Verde\n")
    
    image_path = input("Ingresa la ruta de la imagen con el tapete verde: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ No se encontró la imagen: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error al cargar la imagen")
        return
    
    system = CardRecognitionSystem()
    params = system.calibrate_green_detection(img, show_results=True)
    
    print("\n✅ Calibración completada y guardada")
    print("   Ahora puedes usar el sistema con estas configuraciones")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🧪  SUITE DE TESTS COMPLETA  🧪                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Crear templates de ejemplo si no existen
    if not os.path.exists('templates') or len(os.listdir('templates')) == 0:
        create_template_examples()
    
    # 2. Test de calibración
    test_calibration()
    
    # 3. Mostrar info de templates
    show_template_info()
    
    # 4. Test de carta individual
    test_single_card()
    
    # 5. Test de múltiples cartas
    test_multiple_cards()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              ✅  TODOS LOS TESTS COMPLETADOS  ✅            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main_menu():
    """Menú interactivo principal."""
    while True:
        print("""
┌──────────────────────────────────────────────────────────────┐
│               SISTEMA DE RECONOCIMIENTO DE CARTAS            │
│                         MENÚ PRINCIPAL                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Crear templates de ejemplo                              │
│  2. Mostrar información de templates                        │
│  3. Calibrar tapete verde (interactivo)                     │
│  4. Procesar una imagen                                      │
│  5. Procesar lote de imágenes                               │
│  6. Ejecutar suite completa de tests                        │
│  0. Salir                                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
        """)
        
        choice = input("Selecciona una opción: ").strip()
        
        if choice == '1':
            create_template_examples()
        elif choice == '2':
            show_template_info()
        elif choice == '3':
            interactive_calibration()
        elif choice == '4':
            img_path = input("Ruta de la imagen: ").strip()
            if os.path.exists(img_path):
                system = CardRecognitionSystem()
                system.process_image(img_path)
            else:
                print(f"❌ No se encontró: {img_path}")
        elif choice == '5':
            dir_path = input("Ruta del directorio: ").strip()
            if os.path.exists(dir_path):
                system = CardRecognitionSystem()
                system.batch_process(dir_path)
            else:
                print(f"❌ No se encontró: {dir_path}")
        elif choice == '6':
            run_all_tests()
        elif choice == '0':
            print("\n👋 ¡Hasta luego!\n")
            break
        else:
            print("\n❌ Opción no válida\n")
        
        input("\nPresiona ENTER para continuar...")


if __name__ == "__main__":
    # Puedes descomentar la línea que prefieras:
    
    # Menú interactivo
    # main_menu()
    
    # O ejecutar todos los tests directamente
    run_all_tests()