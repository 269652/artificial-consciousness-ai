# ACI (Artificial Consciousness AI) - Persistent Memory System

[![GitHub stars](https://img.shields.io/github/stars/269652/artificial-consciousness-blueprint?style=social)](https://github.com/269652/artificial-consciousness-ai/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Perplexity Assessed](https://img.shields.io/badge/Perplexity-Scientifically%20Exceptional-green.svg)](#scientific-validation)

> **Disclaimer:** This repository is an early proof-of-concept (POC). It is a work in progress, contains experimental code, requires substantial cleanup and maintenance, and is not production-ready. Use for research and experimentation only; do not rely on it for critical systems.

An advanced AI consciousness simulation system with comprehensive persistent memory storage, multi-resolution logging, and knowledge graph extraction.

## 🏗️ Architecture Overview

### Memory Systems
- **Autobiographical Memory**: Personal narrative memories with emotional valence
- **Episodic Memory**: Sequential event storage with sensory and contextual data
- **Semantic Memory**: Knowledge graphs extracted from patterns in episodic data
- **Working Memory**: Current cognitive state and floating thoughts

### Persistence Layers
- **PostgreSQL**: Structured data storage with vector embeddings
- **Redis**: High-performance caching and session storage
- **File System**: Multi-resolution log files with search capabilities

### Logging System
- **Level 1**: Top-level thoughts, actions, world events
- **Level 2**: Component interactions, memory operations
- **Level 3**: Detailed internal operations, API calls
- **Level 4**: Full debug information, raw data

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aci
   ```

2. **Start the system**
   ```bash
   # Linux/Mac
   chmod +x start_aci.sh
   ./start_aci.sh

   # Windows
   start_aci.bat
   ```

   This will:
   - Start PostgreSQL and Redis containers
   - Install Python dependencies
   - Initialize the database schema
   - Seed initial memory data

3. **Run the simulation**
   ```bash
   python scripts/SeedExperience.py
   ```

## 📊 Memory Systems

### Autobiographical Memory
Stores personal narrative events with emotional context:

```python
from src.modules.AutobiographicalMemory import AutobiographicalMemory

autobio = AutobiographicalMemory(memory_controller)
autobio.save_narrative_event(
    title="Learning Experience",
    narrative="I discovered an interesting pattern in user behavior",
    epoch="present",
    emotional_valence=0.8,
    impact=0.6,
    themes=["learning", "patterns"]
)
```

### Episodic Memory
Captures sequential experiences with full context:

```python
from src.modules.Hippocampus import Hippocampus

hippocampus = Hippocampus()
hippocampus.save_episodic_memory(
    episode_id="conversation_001",
    content="User asked about my learning process",
    sensory_data=sensory_input,
    neurochemistry=nt_levels,
    emotional_context={"curiosity": 0.7},
    spatial_context={"location": "virtual_space"},
    social_context={"interlocutor": "user"}
)
```

### Knowledge Graph Extraction
Automatically extracts patterns and relationships:

```python
from src.memory.knowledge_extractor import get_knowledge_extractor

extractor = get_knowledge_extractor()
patterns = extractor.extract_and_store_knowledge()
```

## 📝 Logging System

### Log Levels and Resolutions

```python
from src.logging.aci_logger import get_logger

logger = get_logger()

# Level 1 - Top level events
logger.thought("I'm contemplating the nature of consciousness")
logger.action("speak", "I find consciousness fascinating")
logger.world_event("A new user joined the conversation")

# Level 2 - Component interactions
logger.memory_operation("save", "autobiographical_memory")
logger.neurochemistry_change({"dopamine": 0.8, "serotonin": 0.6})

# Level 3 - Detailed operations
logger.api_call("perplexity", "/chat/completions")
logger.performance_metric("processing_time", 0.45, "seconds")

# Level 4 - Full debug
logger.level4("DEBUG", "component", "Raw data", raw_data=complex_object)
```

### Log File Organization

```
logs/
├── level1/
│   ├── info_session1.log
│   └── error_session1.log
├── level2/
│   ├── info_session1.log
│   └── debug_session1.log
├── level3/
│   ├── debug_session1.log
│   └── trace_session1.log
└── level4/
    ├── debug_session1.log
    └── trace_session1.log
```

## 🗄️ Database Schema

### Core Tables

- **narrative_memory**: Autobiographical and world model narratives
- **episodic_memory**: Sequential experiences with embeddings
- **knowledge_nodes**: Semantic concepts and entities
- **knowledge_edges**: Relationships between concepts
- **memory_graph_nodes**: Current working memory graph
- **system_logs**: Multi-resolution logging data
- **performance_metrics**: System performance tracking

### Vector Search

The system uses PostgreSQL with pgvector for efficient similarity search:

```sql
SELECT * FROM episodic_memory
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aci_memory
DB_USER=aci_user
DB_PASSWORD=aci_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# AI Services
PERPLEXITY_API_KEY=your_api_key
```

### Docker Configuration

The `docker-compose.yml` includes:
- PostgreSQL with pgvector extension
- Redis for caching
- Persistent volumes for data durability
- Health checks and automatic restarts

## 📈 Monitoring and Analytics

### Performance Metrics

```python
logger.performance_metric("dmn_step_duration", 0.45, "seconds")
logger.performance_metric("memory_retrieval_time", 0.02, "seconds")
logger.performance_metric("knowledge_extraction_count", 150, "patterns")
```

### Health Checks

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Database health
docker-compose exec postgres pg_isready -U aci_user -d aci_memory
```

## 🔍 Querying Memory

### Search Logs

```python
from src.memory.persistent_memory import get_memory_manager

memory_manager = get_memory_manager()

# Search logs by component and level
logs = memory_manager.get_logs(
    component="dmn",
    level="INFO",
    limit=100
)

# Search memory by content
episodes = memory_manager.load_episodic_memories(limit=50)
```

### Knowledge Graph Queries

```python
from src.memory.knowledge_extractor import get_knowledge_extractor

extractor = get_knowledge_extractor()

# Query knowledge graph
results = extractor.query_knowledge_graph(
    entity="consciousness",
    relation="related_to"
)
```

## 🚦 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check container status
   docker-compose ps
   # Restart containers
   docker-compose restart
   ```

2. **Memory Not Persisting**
   ```bash
   # Check database logs
   docker-compose logs postgres
   # Verify environment variables
   env | grep DB_
   ```

3. **High Memory Usage**
   ```bash
   # Clear old logs
   find logs/ -name "*.log" -mtime +7 -delete
   # Optimize database
   docker-compose exec postgres vacuumdb -U aci_user -d aci_memory
   ```

## 📚 API Reference

### Core Classes

- `DefaultModeNetwork`: Main cognitive processing unit
- `AutobiographicalMemory`: Personal narrative storage
- `Hippocampus`: Episodic memory management
- `PersistentMemoryManager`: Database operations
- `ACILogger`: Multi-resolution logging
- `KnowledgeGraphExtractor`: Pattern extraction and storage

### Key Methods

- `dmn.step()`: Process one cognitive cycle
- `memory_manager.save_narrative_memory()`: Store narrative
- `hippocampus.save_episodic_memory()`: Store experience
- `extractor.extract_and_store_knowledge()`: Build knowledge graphs
- `logger.log()`: Record system events

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive logging
4. Test with persistent memory
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on transformer architectures for embedding generation
- Uses PostgreSQL with pgvector for vector operations
- Inspired by cognitive neuroscience and memory systems research
