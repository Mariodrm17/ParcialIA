"""
TEMPLATE MATCHER PROFESIONAL OPTIMIZADO
Comparación píxel a píxel 226x314 sin tirones
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TemplateMatcherPro:
    """Comparador profesional OPTIMIZADO para velocidad"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.card_size = (226, 314)
        self.templates: Dict[str, Dict] = {}
        
        self.suit_names = {
            'CORAZONES': 'Corazones',
            'DIAMANTES': 'Diamantes',
            'PICAS': 'Picas',
            'TREBOLES': 'Tréboles'
        }
        
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Carga plantillas de forma optimizada"""
        print("\n" + "=" * 70)
        print("📁 CARGANDO PLANTILLAS")
        print("=" * 70)
        
        if not self.templates_dir.exists():
            print(f"❌ No existe '{self.templates_dir}'")
            return
        
        total = 0
        
        for suit_folder in sorted(self.templates_dir.iterdir()):
            if not suit_folder.is_dir():
                continue
            
            suit_name = suit_folder.name.upper()
            suit_display = self.suit_names.get(suit_name, suit_name)
            
            print(f"\n📂 {suit_display}:")
            
            for img_path in sorted(suit_folder.glob("*.png")):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Redimensionar SOLO UNA VEZ
                    img_resized = cv2.resize(img, self.card_size, 
                                            interpolation=cv2.INTER_LINEAR)
                    
                    parts = img_path.stem.split('_')
                    number = parts[0]
                    if number == 'AS':
                        number = 'A'
                    
                    card_key = f"{number}_{suit_name}"
                    
                    # PREPROCESAR UNA SOLA VEZ y guardar
                    processed = self._preprocess_once(img_resized)
                    
                    self.templates[card_key] = {
                        'processed': processed,
                        'display_name': f"{number} de {suit_display}",
                        'number': number,
                        'suit': suit_display
                    }
                    
                    print(f"  ✅ {number} de {suit_display}")
                    total += 1
                    
                except Exception as e:
                    print(f"  ❌ {img_path.name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ {total} plantillas cargadas")
        print("=" * 70)
    
    def _preprocess_once(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento RÁPIDO y EFECTIVO
        Solo escala de grises + CLAHE ligero
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE MÁS LIGERO (más rápido)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Normalizar
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def _compare_fast(self, card: np.ndarray, template: np.ndarray) -> float:
        """
        Comparación RÁPIDA pero precisa
        Solo 2 métodos principales
        """
        if card.shape != template.shape:
            template = cv2.resize(template, (card.shape[1], card.shape[0]))
        
        # MÉTODO 1: TM_CCOEFF_NORMED (70% peso)
        result = cv2.matchTemplate(card, template, cv2.TM_CCOEFF_NORMED)
        tm_score = float(result[0][0])
        
        # MÉTODO 2: Correlación normalizada (30% peso)
        card_flat = card.flatten().astype(np.float32)
        template_flat = template.flatten().astype(np.float32)
        
        # Normalizar vectores
        card_norm = (card_flat - np.mean(card_flat)) / (np.std(card_flat) + 1e-7)
        template_norm = (template_flat - np.mean(template_flat)) / (np.std(template_flat) + 1e-7)
        
        # Correlación
        corr = np.dot(card_norm, template_norm) / len(card_norm)
        corr_score = (corr + 1) / 2
        
        # Score final
        final_score = tm_score * 0.7 + corr_score * 0.3
        
        return max(0.0, min(1.0, final_score))
    
    def recognize(self, card_img: np.ndarray, 
                 return_top: int = 3) -> Tuple[Optional[str], float, List[Dict]]:
        """
        Reconocimiento OPTIMIZADO
        """
        if not self.templates:
            return None, 0.0, []
        
        # Redimensionar carta
        card_resized = cv2.resize(card_img, self.card_size, 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Preprocesar (rápido)
        card_processed = self._preprocess_once(card_resized)
        
        # Comparar con TODAS las plantillas
        results = []
        
        for card_key, template_data in self.templates.items():
            score = self._compare_fast(card_processed, template_data['processed'])
            
            results.append({
                'name': template_data['display_name'],
                'score': score,
                'key': card_key
            })
        
        # Ordenar
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            best = results[0]
            return best['name'], best['score'], results[:return_top]
        
        return None, 0.0, []


if __name__ == "__main__":
    matcher = TemplateMatcherPro()
    
    if matcher.templates:
        print(f"\n✅ {len(matcher.templates)} plantillas listas")
    else:
        print("\n❌ No se cargaron plantillas")