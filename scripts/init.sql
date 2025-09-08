-- ACI Memory Database Initialization
-- This script sets up the database schema for persistent memory storage

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For GIN indexes

-- ===========================================
-- NARRATIVE MEMORIES
-- ===========================================

-- Autobiographical Narrative Memory
CREATE TABLE IF NOT EXISTS autobiographical_narrative (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    narrative_type VARCHAR(50) NOT NULL DEFAULT 'autobiographical',
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB,
    neurochemistry JSONB,
    emotional_valence FLOAT,
    importance_score FLOAT DEFAULT 0.5,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- World Model Narrative Memory
CREATE TABLE IF NOT EXISTS world_model_narrative (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    narrative_type VARCHAR(50) NOT NULL DEFAULT 'world_model',
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB,
    neurochemistry JSONB,
    emotional_valence FLOAT,
    importance_score FLOAT DEFAULT 0.5,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Generic Narrative Memory (for extensibility)
CREATE TABLE IF NOT EXISTS narrative_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    narrative_type VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB,
    neurochemistry JSONB,
    emotional_valence FLOAT,
    importance_score FLOAT DEFAULT 0.5,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- EPISODIC MEMORY
-- ===========================================

CREATE TABLE IF NOT EXISTS episodic_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id VARCHAR(100) NOT NULL,
    sequence_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sensory_data JSONB,
    neurochemistry JSONB,
    emotional_context JSONB,
    spatial_context JSONB,
    social_context JSONB,
    thought_chain TEXT[],
    embedding VECTOR(384), -- For similarity search
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(episode_id, sequence_number)
);

-- ===========================================
-- KNOWLEDGE GRAPH MEMORY (Second Layer)
-- ===========================================

-- Knowledge Nodes
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type VARCHAR(50) NOT NULL, -- concept, entity, pattern, etc.
    label VARCHAR(255) NOT NULL,
    description TEXT,
    properties JSONB,
    embedding VECTOR(384),
    confidence_score FLOAT DEFAULT 1.0,
    activation_count INTEGER DEFAULT 0,
    last_activated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Edges (Relationships)
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    properties JSONB,
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_node_id, target_node_id, relation_type)
);

-- ===========================================
-- MEMORY GRAPH (Current Graph Structure)
-- ===========================================

CREATE TABLE IF NOT EXISTS memory_graph_nodes (
    id VARCHAR(255) PRIMARY KEY,
    embedding VECTOR(384),
    tags TEXT[],
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB,
    thought_chain TEXT[],
    neurochemistry JSONB,
    read_count INTEGER DEFAULT 0,
    merge_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_node_id VARCHAR(255) NOT NULL REFERENCES memory_graph_nodes(id) ON DELETE CASCADE,
    to_node_id VARCHAR(255) NOT NULL REFERENCES memory_graph_nodes(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_node_id, to_node_id, relation_type)
);

-- ===========================================
-- LOGGING TABLES
-- ===========================================

CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL, -- ERROR, WARN, INFO, DEBUG, TRACE
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    context JSONB,
    session_id VARCHAR(100),
    step_number INTEGER,
    tags TEXT[]
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(50),
    context JSONB
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- Narrative memory indexes
CREATE INDEX IF NOT EXISTS idx_autobiographical_timestamp ON autobiographical_narrative(timestamp);
CREATE INDEX IF NOT EXISTS idx_autobiographical_tags ON autobiographical_narrative USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_world_model_timestamp ON world_model_narrative(timestamp);
CREATE INDEX IF NOT EXISTS idx_world_model_tags ON world_model_narrative USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_narrative_timestamp ON narrative_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_narrative_type ON narrative_memory(narrative_type);
CREATE INDEX IF NOT EXISTS idx_narrative_tags ON narrative_memory USING GIN(tags);

-- Episodic memory indexes
CREATE INDEX IF NOT EXISTS idx_episodic_episode ON episodic_memory(episode_id);
CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_tags ON episodic_memory USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_episodic_embedding ON episodic_memory USING ivfflat(embedding vector_cosine_ops);

-- Knowledge graph indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_label ON knowledge_nodes(label);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_type ON knowledge_edges(relation_type);

-- Memory graph indexes
CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_timestamp ON memory_graph_nodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_tags ON memory_graph_nodes USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_from ON memory_graph_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_to ON memory_graph_edges(to_node_id);

-- Logging indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_session ON system_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_tags ON system_logs USING GIN(tags);

-- ===========================================
-- FUNCTIONS AND TRIGGERS
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add update triggers to all tables with updated_at
CREATE TRIGGER update_autobiographical_narrative_updated_at BEFORE UPDATE ON autobiographical_narrative FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_world_model_narrative_updated_at BEFORE UPDATE ON world_model_narrative FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_narrative_memory_updated_at BEFORE UPDATE ON narrative_memory FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_episodic_memory_updated_at BEFORE UPDATE ON episodic_memory FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_nodes_updated_at BEFORE UPDATE ON knowledge_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_edges_updated_at BEFORE UPDATE ON knowledge_edges FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_memory_graph_nodes_updated_at BEFORE UPDATE ON memory_graph_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for similarity search on embeddings
CREATE OR REPLACE FUNCTION cosine_similarity(a VECTOR, b VECTOR)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- INITIAL DATA
-- ===========================================

-- Insert default narrative types
INSERT INTO narrative_memory (narrative_type, content, context) VALUES
('autobiographical', 'Initial autobiographical narrative memory', '{"initialized": true}'),
('world_model', 'Initial world model narrative memory', '{"initialized": true}')
ON CONFLICT DO NOTHING;
