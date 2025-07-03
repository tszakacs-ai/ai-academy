"""
Estrazione entità NER semplificata
"""

import re
from typing import List, Dict, Tuple
from transformers import pipeline
import streamlit as st
from dataclasses import dataclass

from config import NER_MODEL, MIN_CONFIDENCE, ENTITY_TYPES, PATTERNS

@dataclass
class Entity:
    """Rappresenta un'entità estratta"""
    text: str           # Testo originale
    label: str          # Tipo NER (PER, ORG, etc.)
    confidence: float   # Score di confidence
    start: int          # Posizione inizio
    end: int            # Posizione fine
    document: str       # Nome documento
    entity_type: str    # Tipo normalizzato

class SimpleNERExtractor:
    """Estrattore NER semplice e comprensibile"""
    
    def __init__(self):
        """Inizializza l'estrattore"""
        self._ner_pipeline = None
        self.entities_found = []
        
    @property 
    def ner_pipeline(self):
        """Carica il modello NER la prima volta che serve (lazy loading)"""
        if self._ner_pipeline is None:
            with st.spinner("🤖 Caricamento modello NER..."):
                try:
                    self._ner_pipeline = pipeline(
                        "ner",
                        model=NER_MODEL,
                        aggregation_strategy="simple"
                    )
                    st.success("✅ Modello NER caricato!")
                except Exception as e:
                    st.error(f"❌ Errore caricamento modello: {e}")
                    return None
        return self._ner_pipeline
    
    def extract_from_text(self, text: str, document_name: str) -> List[Entity]:
        """
        Estrae entità da un singolo testo
        
        Args:
            text: Testo da analizzare
            document_name: Nome del documento
            
        Returns:
            Lista di entità trovate
        """
        entities = []
        
        # Fase 1: Estrazione con NER
        ner_entities = self._extract_with_ner(text, document_name)
        entities.extend(ner_entities)
        
        # Fase 2: Estrazione con pattern regex
        pattern_entities = self._extract_with_patterns(text, document_name)
        entities.extend(pattern_entities)
        
        # Fase 3: Pulizia e normalizzazione
        entities = self._clean_and_normalize(entities)
        
        return entities
    
    def _extract_with_ner(self, text: str, document_name: str) -> List[Entity]:
        """Estrae entità usando il modello NER"""
        entities = []
        
        if not self.ner_pipeline:
            return entities
            
        try:
            # Esegui NER
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                # Filtra per confidence
                if result['score'] >= MIN_CONFIDENCE:
                    entity = Entity(
                        text=result['word'],
                        label=result['entity_group'],
                        confidence=result['score'],
                        start=result['start'],
                        end=result['end'],
                        document=document_name,
                        entity_type=ENTITY_TYPES.get(result['entity_group'], 'OTHER')
                    )
                    entities.append(entity)
                    
        except Exception as e:
            st.warning(f"⚠️ Errore NER per {document_name}: {e}")
            
        return entities
    
    def _extract_with_patterns(self, text: str, document_name: str) -> List[Entity]:
        """Estrae entità usando pattern regex"""
        entities = []
        
        for pattern_type, regex_list in PATTERNS.items():
            for regex_pattern in regex_list:
                # Trova tutte le corrispondenze
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        label=pattern_type,
                        confidence=0.8,  # Confidence fissa per regex
                        start=match.start(),
                        end=match.end(),
                        document=document_name,
                        entity_type=pattern_type
                    )
                    entities.append(entity)
        
        return entities
    
    def _clean_and_normalize(self, entities: List[Entity]) -> List[Entity]:
        """Pulisce e normalizza le entità"""
        cleaned = []
        seen_entities = set()
        
        for entity in entities:
            # Normalizza il testo
            normalized_text = self._normalize_text(entity.text, entity.entity_type)
            
            # Evita duplicati
            entity_key = (normalized_text.lower(), entity.entity_type, entity.document)
            if entity_key not in seen_entities:
                entity.text = normalized_text
                cleaned.append(entity)
                seen_entities.add(entity_key)
        
        return cleaned
    
    def _normalize_text(self, text: str, entity_type: str) -> str:
        """Normalizza il testo dell'entità"""
        text = text.strip()
        
        if entity_type == "PERSON":
            # Rimuovi titoli comuni e normalizza
            titles = ["dott", "prof", "ing", "sig", "sig.ra", "dr", "mr", "mrs"]
            words = text.lower().split()
            filtered = [w for w in words if w not in titles]
            return " ".join(word.capitalize() for word in filtered)
            
        elif entity_type == "ORGANIZATION":
            # Standardizza suffissi aziendali
            replacements = {
                "s.r.l.": "SRL", "s.p.a.": "SPA", 
                "s.s.": "SS", "s.n.c.": "SNC"
            }
            normalized = text
            for old, new in replacements.items():
                normalized = re.sub(old, new, normalized, flags=re.IGNORECASE)
            return normalized.strip()
            
        elif entity_type == "MONEY":
            # Standardizza formato denaro
            return re.sub(r'\s+', '', text)
            
        elif entity_type == "DATE":
            # Standardizza formato date
            return re.sub(r'[^\d\/\-\.]', '', text)
            
        else:
            return text
    
    def extract_from_documents(self, documents: Dict[str, str]) -> List[Entity]:
        """
        Estrae entità da multipli documenti
        
        Args:
            documents: Dict {nome_file: contenuto}
            
        Returns:
            Lista di tutte le entità trovate
        """
        all_entities = []
        
        # Progress bar
        progress_bar = st.progress(0)
        total_docs = len(documents)
        
        for i, (doc_name, content) in enumerate(documents.items()):
            # Aggiorna progress
            progress = (i + 1) / total_docs
            progress_bar.progress(progress, f"📄 Processando: {doc_name}")
            
            # Estrai entità
            entities = self.extract_from_text(content, doc_name)
            all_entities.extend(entities)
            
            # Info per utente
            st.write(f"   📊 {doc_name}: {len(entities)} entità trovate")
        
        progress_bar.empty()
        
        # Salva per statistiche
        self.entities_found = all_entities
        
        return all_entities
    
    def get_statistics(self) -> Dict:
        """Ritorna statistiche sulle entità estratte"""
        if not self.entities_found:
            return {}
        
        # Conta per tipo
        type_counts = {}
        for entity in self.entities_found:
            entity_type = entity.entity_type
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # Conta per documento
        doc_counts = {}
        for entity in self.entities_found:
            doc = entity.document
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        
        # Confidence media
        avg_confidence = sum(e.confidence for e in self.entities_found) / len(self.entities_found)
        
        return {
            "total_entities": len(self.entities_found),
            "by_type": type_counts,
            "by_document": doc_counts,
            "avg_confidence": round(avg_confidence, 3),
            "unique_types": len(type_counts)
        }

# Funzioni helper per uso semplificato

def extract_entities_simple(documents: Dict[str, str]) -> List[Entity]:
    """
    Funzione semplificata per estrarre entità
    
    Args:
        documents: Dict {nome_file: contenuto}
        
    Returns:
        Lista entità estratte
    """
    extractor = SimpleNERExtractor()
    return extractor.extract_from_documents(documents)

def entities_to_dataframe(entities: List[Entity]):
    """Converte lista entità in DataFrame pandas"""
    import pandas as pd
    
    data = []
    for entity in entities:
        data.append({
            "Entità": entity.text,
            "Tipo": entity.entity_type,
            "Confidence": round(entity.confidence, 3),
            "Documento": entity.document,
            "Posizione": f"{entity.start}-{entity.end}"
        })
    
    return pd.DataFrame(data)