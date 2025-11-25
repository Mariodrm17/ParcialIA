"""
TEMPLATE MATCHER - Comparación de Plantillas 226x314
Sistema profesional de template matching píxel por píxel
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TemplateMatcher:
    """Comparador profesional de cartas por template matching"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.card_size = (226, 314)  # Ancho x Alto estándar
        self.templates: Dict[str, Dict] = {}
        
        # Mapeo de nombres
        self.suit_names = {
            'CORAZONES': 'Corazones',
            'DIAMANTES': 'Diamantes',
            'PICAS': 'Picas',
            'TREBOLES': 'Tréboles'
        }
        
        self.number_names = {
            'AS': 'A',
            'J': 'J',
            'Q': 'Q',
            'K': 'K'
        }
        
        # Cargar plantillas
        self.load_templates()
    
    def load_templates(self):
        """Carga todas las plantillas desde la estructura de carpetas"""
        print("=" * 70)
        print("📁 CARGANDO PLANTILLAS")
        print("=" * 70)
        
        if not self.templates_dir.exists():
            print(f"❌ No existe directorio: {self.templates_dir}")
            return
        
        total_loaded = 0
        
        # Recorrer carpetas de palos
        for suit_folder in sorted(self.templates_dir.iterdir()):
            if not suit_folder.is_dir():
                continue
            
            suit_key = suit_folder.name.upper()
            suit_display = self.suit_names.get(suit_key, suit_key)
            
            print(f"\n📂 {suit_display}:")
            
            # Cargar imágenes de este palo
            for img_file in sorted(suit_folder.glob("*.png")):
                try:
                    # Leer imagen
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        print(f"  ⚠️  Error leyendo: {img_file.name}")
                        continue
                    
                    # Redimensionar a 226x314
                    img_resized = cv2.resize(img, self.card_size, 
                                            interpolation=cv2.INTER_AREA)
                    
                    # Extraer número del nombre de archivo
                    # Formato: "2_CORAZONES.png" o "AS_CORAZONES.png"
                    filename_parts = img_file.stem.split('_')
                    number_str = filename_parts[0]
                    
                    # Convertir a formato estándar
                    number = self.number_names.get(number_str, number_str)
                    
                    # Clave única
                    card_key = f"{number}_{suit_key}"
                    
                    # Preprocesar
                    processed = self.preprocess_card(img_resized)
                    
                    # Guardar
                    self.templates[card_key] = {
                        'original': img_resized,
                        'processed': processed,
                        'display_name': f"{number} de {suit_display}",
                        'number': number,
                        'suit': suit_display,
                        'suit_key': suit_key
                    }
                    
                    print(f"  ✅ {number} de {suit_display}")
                    total_loaded += 1
                    
                except Exception as e:
                    print(f"  ❌ Error con {img_file.name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ Total plantillas cargadas: {total_loaded}")
        print(f"🎴 Cartas únicas: {len(self.templates)}")
        print("=" * 70)
    
    def preprocess_card(self, card_img: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento avanzado para normalización
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE para ecualizar histograma adaptativamente
        # Normaliza iluminación en diferentes regiones
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Suavizado para reducir ruido
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Normalizar a rango completo 0-255
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def compare_cards(self, card1: np.ndarray, card2: np.ndarray) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Compara dos cartas usando múltiples métodos
        
        Args:
            card1, card2: Cartas preprocesadas (escala de grises)
            
        Returns:
            (score_final, lista_scores_detallados)
            score_final: 0-1, mayor = más similar
        """
        # Asegurar mismo tamaño
        if card1.shape != card2.shape:
            card2 = cv2.resize(card2, (card1.shape[1], card1.shape[0]))
        
        scores = []
        
        # MÉTODO 1: TM_CCOEFF_NORMED
        # Más robusto a cambios de brillo/contraste
        result = cv2.matchTemplate(card1, card2, cv2.TM_CCOEFF_NORMED)
        score_tm_ccoeff = float(result[0][0])
        scores.append(('TM_CCOEFF', score_tm_ccoeff))
        
        # MÉTODO 2: Diferencia cuadrada media normalizada
        mse = np.mean((card1.astype(float) - card2.astype(float)) ** 2)
        max_mse = 255 * 255  # Máxima diferencia posible
        score_mse = 1.0 - (mse / max_mse)
        scores.append(('MSE', score_mse))
        
        # MÉTODO 3: Correlación normalizada
        card1_norm = (card1 - np.mean(card1)) / (np.std(card1) + 1e-8)
        card2_norm = (card2 - np.mean(card2)) / (np.std(card2) + 1e-8)
        correlation = np.mean(card1_norm * card2_norm)
        score_corr = (correlation + 1) / 2  # Normalizar a [0, 1]
        scores.append(('CORR', score_corr))
        
        # MÉTODO 4: SSIM simplificado
        mean1, mean2 = np.mean(card1), np.mean(card2)
        var1, var2 = np.var(card1), np.var(card2)
        cov = np.mean((card1 - mean1) * (card2 - mean2))
        
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        
        score_ssim = (ssim + 1) / 2  # Normalizar a [0, 1]
        scores.append(('SSIM', score_ssim))
        
        # Promedio ponderado optimizado
        # TM_CCOEFF es el más importante (50%)
        # SSIM captura estructura (25%)
        # MSE y CORR complementarios (12.5% cada uno)
        final_score = (
            score_tm_ccoeff * 0.50 +
            score_ssim * 0.25 +
            score_mse * 0.125 +
            score_corr * 0.125
        )
        
        # Asegurar rango [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, scores
    
    def recognize_card(self, card_img: np.ndarray, 
                      return_top_n: int = 3) -> Tuple[Optional[str], float, List[Dict]]:
        """
        Reconoce una carta comparándola con todas las plantillas
        
        Args:
            card_img: Imagen de la carta (BGR)
            return_top_n: Número de mejores matches a retornar
            
        Returns:
            (mejor_match, confianza, top_n_matches)
        """
        if not self.templates:
            print("⚠️  No hay plantillas cargadas")
            return None, 0.0, []
        
        # Redimensionar a tamaño estándar
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
                'detailed_scores': detailed_scores,
                'number': template_data['number'],
                'suit': template_data['suit']
            })
        
        # Ordenar por score descendente
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Mejor match
        if all_matches:
            best = all_matches[0]
            return best['name'], best['score'], all_matches[:return_top_n]
        
        return None, 0.0, []
    
    def visualize_match(self, card_img: np.ndarray, template_key: str):
        """
        Visualiza comparación entre carta y plantilla (debug)
        """
        if template_key not in self.templates:
            print(f"❌ Plantilla {template_key} no encontrada")
            return
        
        # Preprocesar carta
        card_resized = cv2.resize(card_img, self.card_size)
        card_processed = self.preprocess_card(card_resized)
        
        # Obtener plantilla
        template_data = self.templates[template_key]
        template_processed = template_data['processed']
        
        # Comparar
        score, detailed_scores = self.compare_cards(card_processed, template_processed)
        
        # Calcular diferencia
        diff = cv2.absdiff(card_processed, template_processed)
        
        # Convertir a BGR para visualización
        card_bgr = cv2.cvtColor(card_processed, cv2.COLOR_GRAY2BGR)
        template_bgr = cv2.cvtColor(template_processed, cv2.COLOR_GRAY2BGR)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # Combinar horizontalmente
        comparison = np.hstack([card_bgr, template_bgr, diff_colored])
        
        # Agregar texto con scores
        y = 30
        for method, method_score in detailed_scores:
            text = f"{method}: {method_score:.4f}"
            cv2.putText(comparison, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25
        
        # Score final
        cv2.putText(comparison, f"FINAL: {score:.4f}", (10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Mostrar
        window_name = f"Comparacion: {template_data['display_name']}"
        cv2.imshow(window_name, comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Test del módulo
if __name__ == "__main__":
    print("🧪 TEST: Template Matcher")
    
    matcher = TemplateMatcher()
    
    if not matcher.templates:
        print("\n❌ No hay plantillas")
        print("   Verifica que el directorio 'templates' exista")
        print("   con las carpetas: corazones, diamantes, picas, treboles")
        exit(1)
    
    print("\n✅ Sistema listo para reconocimiento")
    print(f"📊 Plantillas disponibles: {len(matcher.templates)}")
    
    # Listar plantillas
    print("\n📋 Plantillas cargadas:")
    for template_data in sorted(matcher.templates.values(), 
                                key=lambda x: (x['suit'], x['number'])):
        print(f"  • {template_data['display_name']}")