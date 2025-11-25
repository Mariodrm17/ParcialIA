import cv2
import numpy as np
import os
from pathlib import Path

class CardDetectorPro:
    """
    Sistema profesional de detección de cartas
    Template Matching píxel por píxel en 226x314
    """
    
    def __init__(self, templates_dir="templates"):
        self.templates_dir = Path(templates_dir)
        self.card_size = (226, 314)  # Ancho x Alto estándar
        self.templates = {}
        
        # Cargar todas las plantillas
        self.load_all_templates()
    
    def load_all_templates(self):
        """
        Carga todas las plantillas desde la estructura:
        templates/
            corazones/
            diamantes/
            picas/
            treboles/
        """
        print("=" * 70)
        print("📁 CARGANDO PLANTILLAS")
        print("=" * 70)
        
        if not self.templates_dir.exists():
            print(f"❌ No existe el directorio: {self.templates_dir}")
            return
        
        # Mapeo de nombres de archivo a nombres legibles
        suit_names = {
            'CORAZONES': 'Corazones',
            'DIAMANTES': 'Diamantes',
            'PICAS': 'Picas',
            'TREBOLES': 'Tréboles'
        }
        
        number_names = {
            'AS': 'A',
            'J': 'J',
            'Q': 'Q',
            'K': 'K'
        }
        
        total_loaded = 0
        
        # Recorrer cada carpeta de palo
        for suit_folder in self.templates_dir.iterdir():
            if not suit_folder.is_dir():
                continue
            
            suit_key = suit_folder.name.upper()
            suit_display = suit_names.get(suit_key, suit_key)
            
            print(f"\n📂 {suit_display}:")
            
            # Cargar cada imagen en la carpeta
            for img_file in sorted(suit_folder.glob("*.png")):
                try:
                    # Leer imagen
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        print(f"  ⚠️  No se pudo leer: {img_file.name}")
                        continue
                    
                    # Redimensionar a tamaño estándar 226x314
                    img_resized = cv2.resize(img, self.card_size, 
                                            interpolation=cv2.INTER_AREA)
                    
                    # Extraer número/letra del nombre de archivo
                    # Formato: "2_CORAZONES.png" o "AS_CORAZONES.png"
                    filename_parts = img_file.stem.split('_')
                    number_str = filename_parts[0]
                    
                    # Convertir a formato estándar
                    if number_str in number_names:
                        number = number_names[number_str]
                    else:
                        number = number_str
                    
                    # Crear clave única
                    card_key = f"{number}_{suit_key}"
                    
                    # Preprocesar plantilla
                    processed = self.preprocess_card(img_resized)
                    
                    # Guardar
                    self.templates[card_key] = {
                        'original': img_resized,
                        'processed': processed,
                        'display_name': f"{number} de {suit_display}",
                        'number': number,
                        'suit': suit_display
                    }
                    
                    print(f"  ✅ {number} de {suit_display}")
                    total_loaded += 1
                    
                except Exception as e:
                    print(f"  ❌ Error con {img_file.name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ Total plantillas cargadas: {total_loaded}")
        print("=" * 70)
    
    def preprocess_card(self, card_img):
        """
        Preprocesamiento avanzado para normalización
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        # Ecualización de histograma adaptativa (CLAHE)
        # Normaliza la iluminación en diferentes regiones
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Suavizado para reducir ruido
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Normalizar rango 0-255
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def compare_cards(self, card1, card2):
        """
        Compara dos cartas usando múltiples métodos
        Retorna score de similitud (0-1, mayor = más similar)
        """
        # Asegurar mismo tamaño
        if card1.shape != card2.shape:
            card2 = cv2.resize(card2, (card1.shape[1], card1.shape[0]))
        
        scores = []
        
        # MÉTODO 1: Template Matching con TM_CCOEFF_NORMED
        # Es el más robusto para cambios de brillo
        result = cv2.matchTemplate(card1, card2, cv2.TM_CCOEFF_NORMED)
        score_tm = result[0][0]
        scores.append(('TM_CCOEFF', score_tm))
        
        # MÉTODO 2: Diferencia cuadrada media normalizada (MSE)
        mse = np.mean((card1.astype(float) - card2.astype(float)) ** 2)
        # Normalizar: 0 = idénticas, mayor = más diferentes
        # Convertir a score de similitud
        max_mse = 255 * 255  # Máxima diferencia posible
        score_mse = 1.0 - (mse / max_mse)
        scores.append(('MSE', score_mse))
        
        # MÉTODO 3: Correlación normalizada directa
        # Normalizar ambas imágenes a media 0 y std 1
        card1_norm = (card1 - np.mean(card1)) / (np.std(card1) + 1e-8)
        card2_norm = (card2 - np.mean(card2)) / (np.std(card2) + 1e-8)
        
        correlation = np.mean(card1_norm * card2_norm)
        # Convertir de [-1, 1] a [0, 1]
        score_corr = (correlation + 1) / 2
        scores.append(('CORR', score_corr))
        
        # MÉTODO 4: Similitud estructural (SSIM simplificado)
        # Basado en luminancia, contraste y estructura
        mean1 = np.mean(card1)
        mean2 = np.mean(card2)
        var1 = np.var(card1)
        var2 = np.var(card2)
        cov = np.mean((card1 - mean1) * (card2 - mean2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        
        score_ssim = (ssim + 1) / 2  # Normalizar a [0, 1]
        scores.append(('SSIM', score_ssim))
        
        # Promedio ponderado
        # TM_CCOEFF es el más importante (50%)
        # SSIM segundo más importante (25%)
        # MSE y CORR complementarios (12.5% cada uno)
        final_score = (
            score_tm * 0.50 +
            score_ssim * 0.25 +
            score_mse * 0.125 +
            score_corr * 0.125
        )
        
        return final_score, scores
    
    def recognize_card(self, card_img, return_top_n=3):
        """
        Reconoce una carta comparándola con todas las plantillas
        
        Args:
            card_img: Imagen de la carta a reconocer
            return_top_n: Número de mejores coincidencias a retornar
        
        Returns:
            best_match: Nombre de la mejor coincidencia
            confidence: Score de confianza (0-1)
            top_matches: Lista de top N coincidencias con scores
        """
        if not self.templates:
            print("⚠️  No hay plantillas cargadas")
            return None, 0.0, []
        
        # Redimensionar carta a tamaño estándar
        card_resized = cv2.resize(card_img, self.card_size, 
                                 interpolation=cv2.INTER_AREA)
        
        # Preprocesar
        card_processed = self.preprocess_card(card_resized)
        
        # Comparar con todas las plantillas
        all_matches = []
        
        for card_key, template_data in self.templates.items():
            template_processed = template_data['processed']
            
            # Comparar
            score, detailed_scores = self.compare_cards(
                card_processed, 
                template_processed
            )
            
            all_matches.append({
                'name': template_data['display_name'],
                'key': card_key,
                'score': score,
                'detailed_scores': detailed_scores
            })
        
        # Ordenar por score (mayor a menor)
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Mejor coincidencia
        if all_matches:
            best = all_matches[0]
            best_match = best['name']
            confidence = best['score']
            top_matches = all_matches[:return_top_n]
            
            return best_match, confidence, top_matches
        
        return None, 0.0, []
    
    def visualize_comparison(self, card_img, template_key):
        """
        Visualiza la comparación entre una carta y una plantilla
        Útil para debug
        """
        if template_key not in self.templates:
            print(f"❌ Plantilla {template_key} no encontrada")
            return
        
        # Redimensionar carta
        card_resized = cv2.resize(card_img, self.card_size)
        card_processed = self.preprocess_card(card_resized)
        
        # Obtener plantilla
        template_data = self.templates[template_key]
        template_processed = template_data['processed']
        
        # Comparar
        score, detailed_scores = self.compare_cards(
            card_processed, 
            template_processed
        )
        
        # Crear visualización
        # Diferencia absoluta
        diff = cv2.absdiff(card_processed, template_processed)
        
        # Convertir a BGR para mostrar
        card_bgr = cv2.cvtColor(card_processed, cv2.COLOR_GRAY2BGR)
        template_bgr = cv2.cvtColor(template_processed, cv2.COLOR_GRAY2BGR)
        diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # Aplicar colormap a la diferencia (rojo = mucha diferencia)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # Combinar imágenes horizontalmente
        comparison = np.hstack([
            card_bgr,
            template_bgr,
            diff_colored
        ])
        
        # Agregar texto con scores
        y_offset = 20
        for method, method_score in detailed_scores:
            text = f"{method}: {method_score:.4f}"
            cv2.putText(comparison, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # Score final
        cv2.putText(comparison, f"FINAL: {score:.4f}", (10, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Mostrar
        window_name = f"Comparacion: {template_data['display_name']}"
        cv2.imshow(window_name, comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_system():
    """
    Sistema de prueba interactivo
    """
    print("\n" + "=" * 70)
    print("🎯 SISTEMA DE DETECCIÓN DE CARTAS - TEMPLATE MATCHING PRO")
    print("=" * 70)
    
    # Cargar detector
    detector = CardDetectorPro(templates_dir="templates")
    
    if not detector.templates:
        print("\n❌ No se pudieron cargar plantillas")
        print("   Verifica que el directorio 'templates' exista")
        print("   y contenga las carpetas: corazones, diamantes, picas, treboles")
        return
    
    # Cargar calibración del tapete verde
    import json
    
    if not os.path.exists("green_calibration.json"):
        print("\n⚠️  No hay calibración del tapete verde")
        print("   El sistema puede no detectar cartas correctamente")
        print("   Ejecuta: python green_calibrator.py")
        
        usar_sin_calibracion = input("\n¿Continuar sin calibración? (s/n): ")
        if usar_sin_calibracion.lower() != 's':
            return
        
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
    else:
        with open("green_calibration.json", 'r') as f:
            calibration = json.load(f)
        green_lower = np.array(calibration['green_lower'])
        green_upper = np.array(calibration['green_upper'])
        print("✅ Calibración cargada")
    
    # Configurar cámara
    camera_idx = input("\nÍndice de cámara (default=0): ").strip()
    camera_idx = int(camera_idx) if camera_idx else 0
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {camera_idx}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("\n" + "=" * 70)
    print("✅ SISTEMA INICIADO")
    print("=" * 70)
    print("\nControles:")
    print("  q - Salir")
    print("  r - Resetear lista de cartas detectadas")
    print("  s - Guardar cartas detectadas en archivo")
    print("  d - Modo debug (mostrar comparaciones)")
    print("=" * 70)
    
    detected_cards = set()
    debug_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo de la cámara")
            break
        
        display = frame.copy()
        
        # Detectar tapete verde
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Morfología para limpiar
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Invertir: cartas = blanco
        cards_mask = cv2.bitwise_not(green_mask)
        
        # Encontrar contornos de cartas
        contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        cards_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por área (cartas típicas)
            if 2000 < area < 50000:
                # Aproximar a polígono
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buscar rectángulos (4 vértices)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Verificar relación de aspecto típica de cartas
                    if 0.5 <= aspect_ratio <= 0.9:
                        # Extraer carta con margen
                        margin = 20
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        
                        card_img = frame[y1:y2, x1:x2]
                        
                        if card_img.size > 0:
                            cards_count += 1
                            
                            # Reconocer carta
                            card_name, confidence, top_matches = detector.recognize_card(card_img)
                            
                            # Color según confianza
                            if confidence >= 0.8:
                                color = (0, 255, 0)  # Verde - Excelente
                            elif confidence >= 0.65:
                                color = (0, 255, 255)  # Amarillo - Bueno
                            elif confidence >= 0.5:
                                color = (0, 165, 255)  # Naranja - Regular
                            else:
                                color = (0, 0, 255)  # Rojo - Bajo
                            
                            # Dibujar rectángulo
                            cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                            
                            if card_name:
                                # Mostrar nombre y confianza
                                cv2.putText(display, card_name, (x, y-35), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.putText(display, f"{confidence:.1%}", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                # Agregar a detectadas si confianza alta
                                if confidence >= 0.65:
                                    if card_name not in detected_cards:
                                        detected_cards.add(card_name)
                                        print(f"🎴 Nueva carta: {card_name} ({confidence:.1%})")
                                
                                # Modo debug: mostrar top 3
                                if debug_mode and top_matches:
                                    print(f"\n🔍 Top 3 para carta en posición ({x}, {y}):")
                                    for i, match in enumerate(top_matches, 1):
                                        print(f"   {i}. {match['name']}: {match['score']:.4f}")
                            else:
                                cv2.putText(display, "?", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Información en pantalla
        cv2.putText(display, f"Cartas visibles: {cards_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Reconocidas: {len(detected_cards)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "q=Salir | r=Reset | s=Guardar | d=Debug", 
                   (10, display.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar
        cv2.imshow("Detector de Cartas Profesional", display)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            detected_cards.clear()
            print("\n🧹 Lista de cartas reseteda")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"\n🔧 Modo debug: {'ACTIVADO' if debug_mode else 'DESACTIVADO'}")
        elif key == ord('s'):
            if detected_cards:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cartas_detectadas_{timestamp}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("🃏 CARTAS DETECTADAS\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total: {len(detected_cards)}\n\n")
                    
                    for i, card in enumerate(sorted(detected_cards), 1):
                        f.write(f"{i}. {card}\n")
                
                print(f"\n💾 Guardado en: {filename}")
            else:
                print("\n⚠️  No hay cartas para guardar")
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE LA SESIÓN")
    print("=" * 70)
    
    if detected_cards:
        print(f"\nCartas reconocidas: {len(detected_cards)}")
        for i, card in enumerate(sorted(detected_cards), 1):
            print(f"  {i}. {card}")
    else:
        print("\n❌ No se reconocieron cartas")
    
    print("\n" + "=" * 70)
    print("👋 Sistema cerrado")
    print("=" * 70)


if __name__ == "__main__":
    test_system()