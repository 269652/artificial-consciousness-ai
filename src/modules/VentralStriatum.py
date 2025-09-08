from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class VentralStriatum:
    """Ventral Striatum module for analyzing and scoring thought graphs based on NT levels.
    
    Analyzes the enriched thought graph from Hippocampus and assigns scores/tags
    based on neurotransmitter levels to help PFC select optimal thoughts/actions.
    """
    
    def __init__(self):
        # Scoring weights for different NTs and cognitive aspects
        self.scoring_weights = {
            'dopamine': {
                'novelty': 0.8,
                'reward': 0.9,
                'motivation': 0.7,
                'exploration': 0.6
            },
            'serotonin': {
                'safety': 0.8,
                'social': 0.7,
                'mind_wandering': 0.6,
                'stability': 0.5
            },
            'norepinephrine': {
                'attention': 0.9,
                'urgency': 0.8,
                'arousal': 0.7,
                'focus': 0.6
            },
            'oxytocin': {
                'social_bonding': 0.8,
                'empathy': 0.7,
                'trust': 0.6
            },
            'testosterone': {
                'assertiveness': 0.7,
                'drive': 0.8,
                'dominance': 0.6
            }
        }
        
        # Tag categories
        self.tag_categories = {
            'cognitive': ['learning', 'memory', 'reasoning', 'planning'],
            'emotional': ['positive', 'negative', 'neutral', 'anxious'],
            'social': ['cooperative', 'competitive', 'isolated', 'connected'],
            'motivational': ['driven', 'apathetic', 'curious', 'satisfied']
        }
    
    def analyze_thought_graph(self, hippocampus_results: Dict[str, Any], 
                            neurochemistry: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the thought graph from HC and assign scores/tags based on NT levels"""
        
        matches = hippocampus_results.get('matches', [])
        enriched_graph = self._enrich_thought_graph(matches)
        
        # Score each thought/memory based on NT levels
        scored_thoughts = []
        for thought in enriched_graph:
            scores = self._calculate_thought_scores(thought, neurochemistry)
            tags = self._assign_tags(thought, scores)
            
            scored_thought = {
                'thought': thought,
                'scores': scores,
                'tags': tags,
                'overall_score': self._calculate_overall_score(scores)
            }
            scored_thoughts.append(scored_thought)
        
        # Sort by overall score (highest first)
        scored_thoughts.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'scored_thoughts': scored_thoughts,
            'top_thought': scored_thoughts[0] if scored_thoughts else None,
            'analysis_summary': self._generate_analysis_summary(scored_thoughts, neurochemistry)
        }
    
    def _enrich_thought_graph(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich the thought graph with additional context and relationships"""
        enriched = []
        
        for match in matches:
            thought = {
                'text': match.get('text', ''),
                'similarity': match.get('similarity', 0.0),
                'tags': match.get('tags', []),
                'node_id': match.get('node_id', ''),
                # Add enriched context
                'cognitive_load': self._estimate_cognitive_load(match),
                'emotional_valence': self._estimate_emotional_valence(match),
                'social_context': self._estimate_social_context(match),
                'temporal_relevance': self._estimate_temporal_relevance(match)
            }
            enriched.append(thought)
        
        return enriched
    
    def _calculate_thought_scores(self, thought: Dict[str, Any], 
                                neurochemistry: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for a thought based on NT levels and thought characteristics"""
        
        scores = {}
        
        # Base scores from thought characteristics
        base_scores = {
            'novelty': thought.get('cognitive_load', 0.5),
            'reward': thought.get('emotional_valence', 0.5),
            'motivation': thought.get('temporal_relevance', 0.5),
            'exploration': thought.get('similarity', 0.5),
            'safety': 1.0 - thought.get('emotional_valence', 0.5),  # Inverse of negative valence
            'social': thought.get('social_context', 0.5),
            'mind_wandering': 0.5,  # Default
            'stability': thought.get('similarity', 0.5),
            'attention': thought.get('cognitive_load', 0.5),
            'urgency': 1.0 - thought.get('temporal_relevance', 0.5),
            'arousal': thought.get('emotional_valence', 0.5),
            'focus': thought.get('cognitive_load', 0.5),
            'social_bonding': thought.get('social_context', 0.5),
            'empathy': thought.get('social_context', 0.5),
            'trust': thought.get('social_context', 0.5),
            'assertiveness': thought.get('emotional_valence', 0.5),
            'drive': thought.get('temporal_relevance', 0.5),
            'dominance': thought.get('emotional_valence', 0.5)
        }
        
        # Apply NT modulation
        for nt, level in neurochemistry.items():
            if nt in self.scoring_weights:
                nt_weights = self.scoring_weights[nt]
                for aspect, weight in nt_weights.items():
                    if aspect in base_scores:
                        # Modulate base score by NT level and weight
                        modulation = level * weight
                        scores[aspect] = base_scores[aspect] * (1.0 + modulation)
                    else:
                        scores[aspect] = base_scores[aspect]
            else:
                # For NTs without specific weights, use general modulation
                for aspect in base_scores:
                    scores[aspect] = base_scores[aspect] * (1.0 + level * 0.1)
        
        return scores
    
    def _assign_tags(self, thought: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Assign tags to a thought based on its scores"""
        
        tags = []
        
        # Cognitive tags
        if scores.get('attention', 0) > 0.7:
            tags.append('high_attention')
        if scores.get('focus', 0) > 0.7:
            tags.append('focused')
        if scores.get('novelty', 0) > 0.7:
            tags.append('novel')
        
        # Emotional tags
        if scores.get('reward', 0) > 0.7:
            tags.append('rewarding')
        if scores.get('safety', 0) > 0.7:
            tags.append('safe')
        if scores.get('arousal', 0) > 0.8:
            tags.append('arousing')
        
        # Social tags
        if scores.get('social_bonding', 0) > 0.7:
            tags.append('social')
        if scores.get('trust', 0) > 0.7:
            tags.append('trustworthy')
        
        # Motivational tags
        if scores.get('drive', 0) > 0.7:
            tags.append('motivating')
        if scores.get('exploration', 0) > 0.7:
            tags.append('exploratory')
        
        return tags
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall score from individual aspect scores"""
        
        # Weighted combination of key aspects
        weights = {
            'reward': 0.3,
            'motivation': 0.25,
            'attention': 0.2,
            'safety': 0.15,
            'social_bonding': 0.1
        }
        
        overall = 0.0
        total_weight = 0.0
        
        for aspect, weight in weights.items():
            if aspect in scores:
                overall += scores[aspect] * weight
                total_weight += weight
        
        return overall / total_weight if total_weight > 0 else 0.5
    
    def _estimate_cognitive_load(self, match: Dict[str, Any]) -> float:
        """Estimate cognitive load of a thought/memory"""
        text = match.get('text', '')
        if not text:
            return 0.5  # Default cognitive load
        # Simple heuristic: longer text = higher cognitive load
        return min(1.0, len(text.split()) / 100.0)
    
    def _estimate_emotional_valence(self, match: Dict[str, Any]) -> float:
        """Estimate emotional valence (0=negative, 1=positive)"""
        text = match.get('text', '')
        if not text:
            return 0.5  # Neutral
        
        positive_words = ['good', 'happy', 'positive', 'great', 'excellent', 'wonderful']
        negative_words = ['bad', 'sad', 'negative', 'terrible', 'awful', 'horrible']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _estimate_social_context(self, match: Dict[str, Any]) -> float:
        """Estimate social context (0=isolated, 1=social)"""
        text = match.get('text', '')
        if not text:
            return 0.0
        
        social_words = ['person', 'people', 'friend', 'talk', 'conversation', 'social', 'together']
        social_count = sum(1 for word in social_words if word in text.lower())
        
        return min(1.0, social_count / 5.0)
    
    def _estimate_temporal_relevance(self, match: Dict[str, Any]) -> float:
        """Estimate temporal relevance (how recent/important)"""
        # Simple heuristic based on similarity score
        similarity = match.get('similarity', 0.5)
        return similarity
    
    def _generate_analysis_summary(self, scored_thoughts: List[Dict[str, Any]], 
                                 neurochemistry: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of the analysis"""
        
        if not scored_thoughts:
            return {'summary': 'No thoughts to analyze'}
        
        top_thought = scored_thoughts[0]
        
        # Analyze NT influence
        nt_influence = {}
        for nt, level in neurochemistry.items():
            if level > 0.7:
                nt_influence[nt] = 'high'
            elif level > 0.4:
                nt_influence[nt] = 'moderate'
            else:
                nt_influence[nt] = 'low'
        
        return {
            'top_score': top_thought['overall_score'],
            'dominant_tags': top_thought['tags'][:3],  # Top 3 tags
            'nt_influence': nt_influence,
            'thought_count': len(scored_thoughts),
            'summary': f"Analysis complete. Top thought scored {top_thought['overall_score']:.2f} "
                      f"with tags: {', '.join(top_thought['tags'][:3])}"
        }
