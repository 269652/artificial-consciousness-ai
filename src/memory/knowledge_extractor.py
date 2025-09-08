"""
Knowledge Graph Extractor for ACI System
Analyzes episodic memories to extract patterns and build knowledge graphs
"""
from typing import Dict, Any, List, Optional, Tuple, Set
import re
from collections import defaultdict, Counter
from datetime import datetime

from src.memory.persistent_memory import get_memory_manager
from src.logging.aci_logger import get_logger
from src.memory.graph.Thought import Thought

aci_logger = get_logger()

class KnowledgeGraphExtractor:
    """Extracts knowledge graphs from patterns in episodic memories"""

    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d+(?:\.\d+)?\b',  # Numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        self.relation_patterns = [
            (r'is a|are a|was a|were a', 'is_a'),
            (r'part of|belongs to|member of', 'part_of'),
            (r'causes|leads to|results in', 'causes'),
            (r'related to|connected to|linked to', 'related_to'),
            (r'has|have|had', 'has_property'),
            (r'larger than|bigger than|greater than', 'larger_than'),
            (r'smaller than|less than', 'smaller_than'),
        ]

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using pattern matching"""
        entities = set()

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)

        # Filter out common words and single characters
        filtered_entities = []
        for entity in entities:
            if len(entity) > 1 and not entity.lower() in {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}:
                filtered_entities.append(entity)

        return filtered_entities

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations between entities from text"""
        relations = []
        entities = self.extract_entities(text)

        for pattern, relation_type in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find entities before and after the relation
                before_text = text[:match.start()]
                after_text = text[match.end():]

                before_entities = self.extract_entities(before_text)
                after_entities = self.extract_entities(after_text)

                for subject in before_entities[-2:]:  # Last 2 entities before relation
                    for obj in after_entities[:2]:  # First 2 entities after relation
                        relations.append((subject, relation_type, obj))

        return relations

    def analyze_episodic_patterns(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in episodic memories to extract knowledge"""
        aci_logger.level2("INFO", "knowledge_extractor", "Starting pattern analysis",
                         episode_count=len(episodes))

        # Extract entities and relations from all episodes
        all_entities = Counter()
        all_relations = Counter()
        entity_cooccurrences = defaultdict(lambda: defaultdict(int))
        temporal_patterns = defaultdict(list)

        for episode in episodes:
            content = episode.get('content', '')
            timestamp = episode.get('timestamp', datetime.now())

            # Extract entities
            entities = self.extract_entities(content)
            for entity in entities:
                all_entities[entity] += 1

            # Extract relations
            relations = self.extract_relations(content)
            for subject, relation, obj in relations:
                all_relations[(subject, relation, obj)] += 1

            # Track entity co-occurrences
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    entity_cooccurrences[entity1][entity2] += 1
                    entity_cooccurrences[entity2][entity1] += 1

            # Track temporal patterns
            for entity in entities:
                temporal_patterns[entity].append(timestamp)

        # Identify significant patterns
        significant_entities = {entity: count for entity, count in all_entities.items() if count >= 3}
        significant_relations = {rel: count for rel, count in all_relations.items() if count >= 2}
        significant_cooccurrences = {
            (e1, e2): count for e1, e2_dict in entity_cooccurrences.items()
            for e2, count in e2_dict.items() if count >= 2
        }

        patterns = {
            'entities': dict(significant_entities),
            'relations': dict(significant_relations),
            'cooccurrences': dict(significant_cooccurrences),
            'temporal_patterns': dict(temporal_patterns),
            'total_episodes_analyzed': len(episodes)
        }

        aci_logger.level2("INFO", "knowledge_extractor", "Pattern analysis complete",
                         entities_found=len(significant_entities),
                         relations_found=len(significant_relations))

        return patterns

    def build_knowledge_graph(self, patterns: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build knowledge graph from extracted patterns"""
        aci_logger.level2("INFO", "knowledge_extractor", "Building knowledge graph")

        nodes = []
        edges = []

        # Create entity nodes
        for entity, frequency in patterns['entities'].items():
            node_id = self.memory_manager.save_knowledge_node(
                node_type='entity',
                label=entity,
                description=f'Entity appearing in {frequency} episodes',
                properties={'frequency': frequency, 'type': 'entity'},
                embedding=Thought.from_text(f"Entity: {entity}").embedding,
                confidence_score=min(frequency / 10.0, 1.0)
            )

            nodes.append({
                'id': node_id,
                'type': 'entity',
                'label': entity,
                'frequency': frequency
            })

        # Create relation edges
        for (subject, relation, obj), frequency in patterns['relations'].items():
            # Find node IDs for subject and object
            subject_node = next((n for n in nodes if n['label'] == subject), None)
            obj_node = next((n for n in nodes if n['label'] == obj), None)

            if subject_node and obj_node:
                edge_id = self.memory_manager.save_knowledge_edge(
                    source_node_id=subject_node['id'],
                    target_node_id=obj_node['id'],
                    relation_type=relation,
                    weight=frequency / 10.0,
                    properties={'frequency': frequency},
                    confidence_score=min(frequency / 5.0, 1.0)
                )

                edges.append({
                    'id': edge_id,
                    'source': subject_node['id'],
                    'target': obj_node['id'],
                    'relation': relation,
                    'weight': frequency / 10.0
                })

        # Create co-occurrence edges
        for (entity1, entity2), frequency in patterns['cooccurrences'].items():
            entity1_node = next((n for n in nodes if n['label'] == entity1), None)
            entity2_node = next((n for n in nodes if n['label'] == entity2), None)

            if entity1_node and entity2_node:
                edge_id = self.memory_manager.save_knowledge_edge(
                    source_node_id=entity1_node['id'],
                    target_node_id=entity2_node['id'],
                    relation_type='co_occurs_with',
                    weight=frequency / 5.0,
                    properties={'cooccurrence_frequency': frequency},
                    confidence_score=min(frequency / 3.0, 1.0)
                )

                edges.append({
                    'id': edge_id,
                    'source': entity1_node['id'],
                    'target': entity2_node['id'],
                    'relation': 'co_occurs_with',
                    'weight': frequency / 5.0
                })

        aci_logger.level2("INFO", "knowledge_extractor", "Knowledge graph built",
                         nodes_created=len(nodes), edges_created=len(edges))

        return nodes, edges

    def extract_and_store_knowledge(self, episodes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main method to extract knowledge from episodes and store in database"""
        try:
            if episodes is None:
                # Load recent episodes from database
                episodes = self.memory_manager.load_episodic_memories(limit=1000)

            # Analyze patterns
            patterns = self.analyze_episodic_patterns(episodes)

            # Build knowledge graph
            nodes, edges = self.build_knowledge_graph(patterns)

            result = {
                'patterns_analyzed': patterns,
                'nodes_created': len(nodes),
                'edges_created': len(edges),
                'episodes_processed': len(episodes)
            }

            aci_logger.level1("INFO", "knowledge_extractor", "Knowledge extraction complete",
                             **result)

            return result

        except Exception as e:
            aci_logger.error(f"Failed to extract and store knowledge: {e}",
                           component="knowledge_extractor")
            return {'error': str(e)}

    def query_knowledge_graph(self, entity: str = None, relation: str = None,
                            limit: int = 50) -> Dict[str, Any]:
        """Query the knowledge graph for specific patterns"""
        try:
            nodes, edges = self.memory_manager.load_knowledge_graph()

            # Filter nodes
            filtered_nodes = nodes
            if entity:
                filtered_nodes = [n for n in nodes if entity.lower() in n.get('label', '').lower()]

            # Filter edges
            filtered_edges = edges
            if relation:
                filtered_edges = [e for e in edges if relation.lower() in e.get('relation_type', '').lower()]

            # Get connected nodes for filtered edges
            connected_node_ids = set()
            for edge in filtered_edges:
                connected_node_ids.add(edge.get('source_node_id'))
                connected_node_ids.add(edge.get('target_node_id'))

            connected_nodes = [n for n in nodes if n.get('id') in connected_node_ids]

            return {
                'nodes': filtered_nodes[:limit],
                'edges': filtered_edges[:limit],
                'connected_nodes': connected_nodes[:limit],
                'total_nodes': len(filtered_nodes),
                'total_edges': len(filtered_edges)
            }

        except Exception as e:
            aci_logger.error(f"Failed to query knowledge graph: {e}",
                           component="knowledge_extractor")
            return {'error': str(e)}

# Global instance
_knowledge_extractor = None

def get_knowledge_extractor() -> KnowledgeGraphExtractor:
    """Get global knowledge extractor instance"""
    global _knowledge_extractor
    if _knowledge_extractor is None:
        _knowledge_extractor = KnowledgeGraphExtractor()
    return _knowledge_extractor
