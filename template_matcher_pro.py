"""
TEMPLATE MATCHER PROFESIONAL
Sistema de reconocimiento por comparación píxel a píxel 226x314
Optimizado para máxima precisión
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

class TemplateMatcherPro:
    """
    Comparador profesional de cartas
    - Tamaño estándar: 226x314
    - Múltiples métodos de comparación
    - Normalización avanzada
    """
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.card_size = (226, 314)  # Ancho x Alto
        self.templates: Dict[str, Dict] = {}
        
        # Nombres en español
        self.suit_names = {
            'CORAZONES': 'Corazones',
            'DIAMANTES': 'Diamantes',
            'PICAS': 'Picas',
            'TREBOLES': 'Tréboles'
        }
        
        # Cargar plantillas
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Carga todas las plantillas desde carpetas"""
        print("\n" + "=" * 70)
        print("📁 CARGANDO PLANTILLAS DE CARTAS")
        print("=" * 70)
        
        if not self.templates_dir.exists():
            print(f"❌ ERROR: No existe el directorio '{self.templates_dir}'")
            return
        
        total = 0
        
        # Recorrer cada carpeta de palo
        for suit_folder in sorted(self.templates_dir.iterdir()):
            if not suit_folder.is_dir():
                continue
            
            suit_name = suit_folder.name.upper()
            suit_display = self.suit_names.get(suit_name, suit_name)
            
            print(f"\n📂 {suit_display}:")
            
            # Cargar cada imagen
            for img_path in sorted(suit_folder.glob("*.png")):
                try:
                    # Leer imagen
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Redimensionar a tamaño exacto 226x314
                    img_resized = cv2.resize(img, self.card_size, 
                                            interpolation=cv2.INTER_CUBIC)
                    
                    # Extraer número del nombre de archivo
                    # Formato: "2_CORAZONES.png", "AS_CORAZONES.png", etc.
                    parts = img_path.stem.split('_')
                    number = parts[0]
                    
                    # Normalizar nombres de números/figuras
                    if number == 'AS':
                        number = 'A'
                    
                    # Clave única
                    card_key = f"{number}_{suit_name}"
                    
                    # Preprocesar (múltiples versiones para mejor matching)
                    processed_versions = self._preprocess_template(img_resized)
                    
                    # Guardar
                    self.templates[card_key] = {
                        'original': img_resized,
                        'processed': processed_versions,
                        'display_name': f"{number} de {suit_display}",
                        'number': number,
                        'suit': suit_display
                    }
                    
                    print(f"  ✅ {number} de {suit_display}")
                    total += 1
                    
                except Exception as e:
                    print(f"  ❌ Error con {img_path.name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ TOTAL CARGADO: {total} plantillas")
        print("=" * 70)
    
    def _preprocess_template(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocesa plantilla con múltiples técnicas
        Retorna diccionario con diferentes versiones
        """
        # Versión 1: Escala de grises con CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # Versión 2: Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Versión 3: Edges (Canny)
        edges = cv2.Canny(gray_clahe, 50, 150)
        
        # Versión 4: Normalizada
        normalized = cv2.normalize(gray_clahe, None, 0, 255, cv2.NORM_MINMAX)
        
        return {
            'gray_clahe': gray_clahe,
            'binary': binary,
            'edges': edges,
            'normalized': normalized
        }
    
    def _preprocess_card(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocesa carta detectada (igual que plantillas)"""
        return self._preprocess_template(img)
    
    def _compare_images(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compara dos imágenes con múltiples métodos
        Retorna score 0-1 (1 = idénticas)
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        scores = []
        
        # MÉTODO 1: Template Matching (TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        tm_score = float(result[0][0])
        scores.append(tm_score * 0.4)  # 40% peso
        
        # MÉTODO 2: Correlación normalizada
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-7)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-7)
        corr = np.mean(img1_norm * img2_norm)
        corr_score = (corr + 1) / 2
        scores.append(corr_score * 0.3)  # 30% peso
        
        # MÉTODO 3: SSIM simplificado
        mean1, mean2 = np.mean(img1), np.mean(img2)
        var1, var2 = np.var(img1), np.var(img2)
        cov = np.mean((img1 - mean1) * (img2 - mean2))
        
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        ssim = ((2*mean1*mean2 + c1) * (2*cov + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        ssim_score = (ssim + 1) / 2
        scores.append(ssim_score * 0.2)  # 20% peso
        
        # MÉTODO 4: Diferencia absoluta invertida
        diff = cv2.absdiff(img1, img2)
        diff_score = 1.0 - (np.mean(diff) / 255.0)
        scores.append(diff_score * 0.1)  # 10% peso
        
        return sum(scores)
    
    def recognize(self, card_img: np.ndarray, 
                 return_top: int = 3) -> Tuple[Optional[str], float, List[Dict]]:
        """
        Reconoce carta comparándola con todas las plantillas
        
        Args:
            card_img: Imagen de la carta (BGR)
            return_top: Número de mejores matches
            
        Returns:
            (nombre_carta, confianza, top_matches)
        """
        if not self.templates:
            return None, 0.0, []
        
        # Redimensionar a tamaño estándar
        card_resized = cv2.resize(card_img, self.card_size, 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Preprocesar carta
        card_processed = self._preprocess_card(card_resized)
        
        # Comparar con todas las plantillas
        results = []
        
        for card_key, template_data in self.templates.items():
            template_processed = template_data['processed']
            
            # Comparar todas las versiones y usar mejor score
            scores_versions = []
            
            for version_name in ['gray_clahe', 'normalized']:
                score = self._compare_images(
                    card_processed[version_name],
                    template_processed[version_name]
                )
                scores_versions.append(score)
            
            # Mejor score entre versiones
            best_score = max(scores_versions)
            
            results.append({
                'name': template_data['display_name'],
                'score': best_score,
                'key': card_key
            })
        
        # Ordenar por score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Mejor resultado
        if results:
            best = results[0]
            return best['name'], best['score'], results[:return_top]
        
        return None, 0.0, []


# Test rápido
if __name__ == "__main__":
    print("\n🧪 TEST: Template Matcher Pro")
    
    matcher = TemplateMatcherPro()
    
    if matcher.templates:
        print(f"\n✅ Sistema listo con {len(matcher.templates)} plantillas")
    else:
        print("\n❌ No se cargaron plantillas")
        print("   Verifica que exista el directorio 'templates/'")
        print("   con carpetas: corazones, diamantes, picas, treboles")