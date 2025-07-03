"""
Costruzione grafi di conoscenza semplificata
"""

import re
import json
from typing import List, Dict, Tuple
import pandas as pd
import networkx as nx
from dataclasses import dataclass

from ner_extractor import Entity
from config import ENTITY_COLORS

@dataclass
class Relationship:
    """Rappresenta una relazione tra entità"""
    source: str         # Entità sorgente
    target: str         # Entità destinazione  
    relation_type: str  # Tipo relazione
    confidence: float   # Confidence della relazione
    context: str        # Contesto testuale
    document: str       # Documento origine

class SimpleGraphBuilder:
    """Costruttore di grafi semplice e comprensibile"""
    
    def __init__(self):
        """Inizializza il costruttore"""
        self.entities = []
        self.relationships = []
        self.graph = nx.Graph()  # Grafo non direzionale per semplicità
    
    def add_entities(self, entities: List[Entity]):
        """
        Aggiunge entità al grafo
        
        Args:
            entities: Lista di entità da aggiungere
        """
        self.entities = entities
        
        # Raggruppa entità per nome normalizzato
        entity_groups = {}
        for entity in entities:
            key = entity.text.lower().strip()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Aggiungi nodi al grafo
        for entity_name, entity_list in entity_groups.items():
            # Prendi la prima entità come rappresentativa
            main_entity = entity_list[0]
            
            # Raccogli documenti dove appare
            documents = list(set([e.document for e in entity_list]))
            
            # Aggiungi nodo
            self.graph.add_node(
                main_entity.text,
                entity_type=main_entity.entity_type,
                confidence=main_entity.confidence,
                documents=documents,
                count=len(entity_list)
            )
    
    def find_relationships(self, documents: Dict[str, str]):
        """
        Trova relazioni tra entità nei documenti
        
        Args:
            documents: Dict {nome_file: contenuto_testo}
        """
        for doc_name, content in documents.items():
            # Trova entità in questo documento
            doc_entities = [e for e in self.entities if e.document == doc_name]
            
            # Cerca relazioni tra coppie di entità
            for i, entity1 in enumerate(doc_entities):
                for entity2 in doc_entities[i+1:]:
                    relationship = self._detect_relationship(
                        entity1, entity2, content, doc_name
                    )
                    if relationship:
                        self.relationships.append(relationship)
                        
                        # Aggiungi arco al grafo
                        self.graph.add_edge(
                            entity1.text,
                            entity2.text,
                            relation_type=relationship.relation_type,
                            confidence=relationship.confidence,
                            context=relationship.context,
                            document=doc_name
                        )
    
    def _detect_relationship(self, entity1: Entity, entity2: Entity, 
                           text: str, document: str) -> Relationship:
        """
        Rileva relazione tra due entità
        
        Args:
            entity1, entity2: Le due entità
            text: Testo del documento
            document: Nome documento
            
        Returns:
            Relationship se trovata, None altrimenti
        """
        # Calcola distanza nel testo
        distance = abs(entity1.start - entity2.start)
        
        # Se troppo distanti, probabilmente non correlate
        if distance > 200:  # Più di 200 caratteri
            return None
        
        # Estrai contesto intorno alle entità
        start_pos = min(entity1.start, entity2.start) - 50
        end_pos = max(entity1.end, entity2.end) + 50
        start_pos = max(0, start_pos)
        end_pos = min(len(text), end_pos)
        context = text[start_pos:end_pos]
        
        # Rileva tipo di relazione
        relation_type = self._classify_relationship(entity1, entity2, context)
        
        if relation_type:
            return Relationship(
                source=entity1.text,
                target=entity2.text,
                relation_type=relation_type,
                confidence=0.7,  # Confidence base
                context=context.strip(),
                document=document
            )
        
        return None
    
    def _classify_relationship(self, entity1: Entity, entity2: Entity, context: str) -> str:
        """
        Classifica il tipo di relazione tra due entità
        
        Args:
            entity1, entity2: Le due entità
            context: Contesto testuale
            
        Returns:
            Tipo di relazione o None
        """
        context_lower = context.lower()
        
        # Relazione LAVORA_PER (Persona -> Organizzazione)
        if (entity1.entity_type == "PERSON" and entity2.entity_type == "ORGANIZATION") or \
           (entity1.entity_type == "ORGANIZATION" and entity2.entity_type == "PERSON"):
            
            work_keywords = ["lavora", "impiegato", "dipendente", "assume", "contratto", 
                           "collabora", "consulente", "direttore", "manager"]
            
            if any(keyword in context_lower for keyword in work_keywords):
                return "LAVORA_PER"
        
        # Relazione PAGA (con importo)
        if entity1.entity_type == "MONEY" or entity2.entity_type == "MONEY":
            payment_keywords = ["paga", "pagamento", "bonifico", "fattura", "importo", 
                               "dovuto", "versamento", "compenso"]
            
            if any(keyword in context_lower for keyword in payment_keywords):
                return "PAGAMENTO"
        
        # Relazione UBICAZIONE
        if entity1.entity_type == "LOCATION" or entity2.entity_type == "LOCATION":
            location_keywords = ["sede", "via", "corso", "piazza", "strada", "ubicato", 
                                "indirizzo", "situato"]
            
            if any(keyword in context_lower for keyword in location_keywords):
                return "UBICAZIONE"
        
        # Relazione TEMPORALE
        if entity1.entity_type == "DATE" or entity2.entity_type == "DATE":
            time_keywords = ["data", "scadenza", "termine", "inizio", "fine", "periodo"]
            
            if any(keyword in context_lower for keyword in time_keywords):
                return "TEMPORALE"
        
        # Relazione COMUNICAZIONE
        if entity1.entity_type == "EMAIL" or entity2.entity_type == "EMAIL":
            return "COMUNICAZIONE"
        
        # Relazione generica ASSOCIATO (se entità sono vicine)
        return "ASSOCIATO"
    
    def build_graph(self, entities: List[Entity], documents: Dict[str, str]):
        """
        Costruisce il grafo completo
        
        Args:
            entities: Lista entità estratte
            documents: Documenti originali
        """
        # Reset grafo
        self.graph.clear()
        
        # Aggiungi entità
        self.add_entities(entities)
        
        # Trova relazioni
        self.find_relationships(documents)
    
    def get_graph_stats(self) -> Dict:
        """Ritorna statistiche del grafo"""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}
        
        # Statistiche base
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 3)
        }
        
        # Conta tipi entità
        entity_types = {}
        for node, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get('entity_type', 'OTHER')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        stats["entity_types"] = entity_types
        
        # Conta tipi relazioni
        relation_types = {}
        for u, v, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relation_type', 'UNKNOWN')
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        stats["relation_types"] = relation_types
        
        # Nodi più connessi
        degrees = dict(self.graph.degree())
        if degrees:
            most_connected = max(degrees, key=degrees.get)
            stats["most_connected"] = {
                "node": most_connected,
                "connections": degrees[most_connected]
            }
        
        return stats
    
    def export_to_json(self) -> str:
        """Esporta grafo in formato JSON"""
        data = {
            "nodes": [
                {
                    "id": node,
                    "label": node,
                    **attrs
                }
                for node, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **attrs
                }
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "statistics": self.get_graph_stats()
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    
    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converte grafo in DataFrame"""
        # DataFrame entità
        entities_data = []
        for node, attrs in self.graph.nodes(data=True):
            entities_data.append({
                "Entità": node,
                "Tipo": attrs.get('entity_type', 'OTHER'),
                "Confidence": attrs.get('confidence', 0),
                "Documenti": ', '.join(attrs.get('documents', [])),
                "Occorrenze": attrs.get('count', 1)
            })
        entities_df = pd.DataFrame(entities_data)
        
        # DataFrame relazioni
        relations_data = []
        for u, v, attrs in self.graph.edges(data=True):
            relations_data.append({
                "Da": u,
                "A": v,
                "Tipo Relazione": attrs.get('relation_type', 'UNKNOWN'),
                "Confidence": attrs.get('confidence', 0),
                "Documento": attrs.get('document', ''),
                "Contesto": attrs.get('context', '')[:100] + "..." if len(attrs.get('context', '')) > 100 else attrs.get('context', '')
            })
        relations_df = pd.DataFrame(relations_data)
        
        return entities_df, relations_df

# Funzioni helper per uso semplificato

def build_knowledge_graph(entities: List[Entity], documents: Dict[str, str]) -> SimpleGraphBuilder:
    """
    Funzione semplificata per costruire grafo
    
    Args:
        entities: Lista entità estratte
        documents: Documenti originali
        
    Returns:
        Grafo costruito
    """
    builder = SimpleGraphBuilder()
    builder.build_graph(entities, documents)
    return builder