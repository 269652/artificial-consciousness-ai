"""
Persistent Memory Layer for ACI System
Handles database operations for all memory types and logging
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import redis

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager with connection pooling"""

    def __init__(self):
        self.pool = None
        self.redis_client = None
        self._init_connections()

    def _init_connections(self):
        """Initialize database and Redis connections"""
        try:
            # PostgreSQL connection
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'aci_memory'),
                'user': os.getenv('DB_USER', 'aci_user'),
                'password': os.getenv('DB_PASSWORD', 'aci_password'),
                'minconn': 1,
                'maxconn': 10
            }

            self.pool = SimpleConnectionPool(**db_config)
            logger.info("PostgreSQL connection pool initialized")

            # Redis connection
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            self.redis_client.ping()  # Test connection
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)

    def close(self):
        """Close all connections"""
        if self.pool:
            self.pool.closeall()
        if self.redis_client:
            self.redis_client.close()

class PersistentMemoryManager:
    """Main manager for all persistent memory operations"""

    def __init__(self):
        self.db = DatabaseConnection()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_narrative_memory(self, narrative_type: str, content: str,
                            context: Dict[str, Any] = None,
                            neurochemistry: Dict[str, float] = None,
                            emotional_valence: float = 0.0,
                            importance_score: float = 0.5,
                            tags: List[str] = None) -> str:
        """Save narrative memory to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO narrative_memory
                (narrative_type, content, context, neurochemistry,
                 emotional_valence, importance_score, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """
                cursor.execute(query, (
                    narrative_type, content,
                    json.dumps(context or {}),
                    json.dumps(neurochemistry or {}),
                    emotional_valence, importance_score,
                    tags or []
                ))
                memory_id = cursor.fetchone()[0]
                conn.commit()
                return str(memory_id)

    def load_narrative_memories(self, narrative_type: str = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Load narrative memories from database"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if narrative_type:
                    query = """
                    SELECT * FROM narrative_memory
                    WHERE narrative_type = %s
                    ORDER BY timestamp DESC LIMIT %s
                    """
                    cursor.execute(query, (narrative_type, limit))
                else:
                    query = """
                    SELECT * FROM narrative_memory
                    ORDER BY timestamp DESC LIMIT %s
                    """
                    cursor.execute(query, (limit,))

                results = cursor.fetchall()
                return [dict(row) for row in results]

    def save_episodic_memory(self, episode_id: str, sequence_number: int,
                           content: str, sensory_data: Dict[str, Any] = None,
                           neurochemistry: Dict[str, float] = None,
                           emotional_context: Dict[str, Any] = None,
                           spatial_context: Dict[str, Any] = None,
                           social_context: Dict[str, Any] = None,
                           thought_chain: List[str] = None,
                           embedding: List[float] = None,
                           tags: List[str] = None) -> str:
        """Save episodic memory to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO episodic_memory
                (episode_id, sequence_number, content, sensory_data,
                 neurochemistry, emotional_context, spatial_context,
                 social_context, thought_chain, embedding, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (episode_id, sequence_number)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    sensory_data = EXCLUDED.sensory_data,
                    neurochemistry = EXCLUDED.neurochemistry,
                    emotional_context = EXCLUDED.emotional_context,
                    spatial_context = EXCLUDED.spatial_context,
                    social_context = EXCLUDED.social_context,
                    thought_chain = EXCLUDED.thought_chain,
                    embedding = EXCLUDED.embedding,
                    tags = EXCLUDED.tags,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """
                cursor.execute(query, (
                    episode_id, sequence_number, content,
                    json.dumps(sensory_data or {}),
                    json.dumps(neurochemistry or {}),
                    json.dumps(emotional_context or {}),
                    json.dumps(spatial_context or {}),
                    json.dumps(social_context or {}),
                    thought_chain or [],
                    embedding,
                    tags or []
                ))
                memory_id = cursor.fetchone()[0]
                conn.commit()
                return str(memory_id)

    def load_episodic_memories(self, episode_id: str = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Load episodic memories from database"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if episode_id:
                    query = """
                    SELECT * FROM episodic_memory
                    WHERE episode_id = %s
                    ORDER BY sequence_number ASC
                    """
                    cursor.execute(query, (episode_id,))
                else:
                    query = """
                    SELECT * FROM episodic_memory
                    ORDER BY timestamp DESC LIMIT %s
                    """
                    cursor.execute(query, (limit,))

                results = cursor.fetchall()
                return [dict(row) for row in results]

    def save_knowledge_node(self, node_type: str, label: str,
                          description: str = "", properties: Dict[str, Any] = None,
                          embedding: List[float] = None,
                          confidence_score: float = 1.0) -> str:
        """Save knowledge node to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO knowledge_nodes
                (node_type, label, description, properties, embedding, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """
                cursor.execute(query, (
                    node_type, label, description,
                    json.dumps(properties or {}),
                    embedding, confidence_score
                ))
                result = cursor.fetchone()
                if result:
                    node_id = str(result[0])
                    conn.commit()
                    return node_id
                else:
                    # Node already exists, get its ID
                    query = "SELECT id FROM knowledge_nodes WHERE label = %s AND node_type = %s"
                    cursor.execute(query, (label, node_type))
                    result = cursor.fetchone()
                    return str(result[0]) if result else None

    def save_knowledge_edge(self, source_node_id: str, target_node_id: str,
                          relation_type: str, weight: float = 1.0,
                          properties: Dict[str, Any] = None,
                          confidence_score: float = 1.0) -> str:
        """Save knowledge edge to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO knowledge_edges
                (source_node_id, target_node_id, relation_type, weight, properties, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_node_id, target_node_id, relation_type)
                DO UPDATE SET
                    weight = EXCLUDED.weight,
                    properties = EXCLUDED.properties,
                    confidence_score = EXCLUDED.confidence_score,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """
                cursor.execute(query, (
                    source_node_id, target_node_id, relation_type,
                    weight, json.dumps(properties or {}), confidence_score
                ))
                edge_id = cursor.fetchone()[0]
                conn.commit()
                return str(edge_id)

    def load_knowledge_graph(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load complete knowledge graph from database"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Load nodes
                cursor.execute("SELECT * FROM knowledge_nodes ORDER BY created_at")
                nodes = [dict(row) for row in cursor.fetchall()]

                # Load edges
                cursor.execute("SELECT * FROM knowledge_edges ORDER BY created_at")
                edges = [dict(row) for row in cursor.fetchall()]

                return nodes, edges

    def save_memory_graph_node(self, node_id: str, embedding: List[float] = None,
                             tags: List[str] = None, timestamp: datetime = None,
                             context: Dict[str, Any] = None,
                             thought_chain: List[str] = None,
                             neurochemistry: Dict[str, float] = None) -> bool:
        """Save memory graph node to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO memory_graph_nodes
                (id, embedding, tags, timestamp, context, thought_chain, neurochemistry)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    tags = EXCLUDED.tags,
                    timestamp = EXCLUDED.timestamp,
                    context = EXCLUDED.context,
                    thought_chain = EXCLUDED.thought_chain,
                    neurochemistry = EXCLUDED.neurochemistry,
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(query, (
                    node_id, embedding, tags or [],
                    timestamp or datetime.now(),
                    json.dumps(context or {}),
                    thought_chain or [],
                    json.dumps(neurochemistry or {})
                ))
                conn.commit()
                return True

    def save_memory_graph_edge(self, from_node_id: str, to_node_id: str,
                             relation_type: str, weight: float = 1.0) -> bool:
        """Save memory graph edge to database"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO memory_graph_edges
                (from_node_id, to_node_id, relation_type, weight)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (from_node_id, to_node_id, relation_type) DO NOTHING
                """
                cursor.execute(query, (from_node_id, to_node_id, relation_type, weight))
                conn.commit()
                return True

    def load_memory_graph(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load complete memory graph from database"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Load nodes
                cursor.execute("SELECT * FROM memory_graph_nodes ORDER BY timestamp")
                nodes = [dict(row) for row in cursor.fetchall()]

                # Load edges
                cursor.execute("SELECT * FROM memory_graph_edges ORDER BY created_at")
                edges = [dict(row) for row in cursor.fetchall()]

                return nodes, edges

    def log_event(self, level: str, component: str, message: str,
                 context: Dict[str, Any] = None, step_number: int = None,
                 tags: List[str] = None):
        """Log event to database and file"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO system_logs
                (level, component, message, context, session_id, step_number, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    level, component, message,
                    json.dumps(context or {}),
                    self.session_id, step_number,
                    tags or []
                ))
                conn.commit()

    def get_logs(self, level: str = None, component: str = None,
                session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve logs from database"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                conditions = []
                params = []

                if level:
                    conditions.append("level = %s")
                    params.append(level)
                if component:
                    conditions.append("component = %s")
                    params.append(component)
                if session_id:
                    conditions.append("session_id = %s")
                    params.append(session_id)

                where_clause = " AND ".join(conditions) if conditions else "TRUE"

                query = f"""
                SELECT * FROM system_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC LIMIT %s
                """
                params.append(limit)
                cursor.execute(query, params)

                results = cursor.fetchall()
                return [dict(row) for row in results]

    def close(self):
        """Close database connections"""
        self.db.close()

# Global instance
_memory_manager = None

def get_memory_manager() -> PersistentMemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = PersistentMemoryManager()
    return _memory_manager
