import logging
import time
from typing import List
from src.memory.graph.controller import MemoryGraphController
from src.memory.graph.node import Node
from src.memory.graph.Thought import Thought

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("memory_test")

SIMILAR_GROUPS = [
    ["The red fox jumps over the sleeping dog", "A red fox leaps above a tired canine", "A quick fox vaults a sleepy hound"],
    ["Quantum particles exhibit wave behavior", "Subatomic entities show wave-like properties"],
    ["The neural network improves with more data", "Training data volume enhances model learning"],
]
DIVERSE_THOUGHTS = [
    "Rain falls gently on the silent lake",
    "A spaceship approaches a blue gas giant",
    "Fresh coffee aroma fills the bustling cafe",
    "Ancient ruins crumble under desert winds",
    "A pianist performs a haunting nocturne",
    "Glaciers retreat under warming climate",
    "A child learns to ride a bicycle",
    "Bioluminescent algae glow at night",
    "Chess grandmaster contemplates a sacrifice",
    "A chef plates a minimalist dish with precision",
]


def build_memories() -> List[Node]:
    controller = MemoryGraphController()
    nodes: List[Node] = []

    # Create similar groups first to ensure similarity edges
    idx = 0
    for group in SIMILAR_GROUPS:
        base_thoughts = [Thought.from_text(t) for t in group]
        for th in base_thoughts:
            node = Node(node_id=f"n{idx}", embedding=th.embedding, tags=["group_sim"], context={
                "thought": th.text,
                "neurochemistry": {"dopamine": 0.1 * (idx % 5), "serotonin": 0.05 * (idx % 3)},
                "scene": f"scene_{idx%4}",
            })
            controller.insert_node_with_relations(node)
            nodes.append(node)
            idx += 1

    # Add diverse thoughts
    for t in DIVERSE_THOUGHTS:
        th = Thought.from_text(t)
        node = Node(node_id=f"n{idx}", embedding=th.embedding, tags=["diverse", f"topic{idx%5}"], context={
            "thought": th.text,
            "neurochemistry": {"dopamine": 0.05 * (idx % 4), "serotonin": 0.07 * (idx % 2)},
            "scene": f"scene_{idx%3}",
        })
        controller.insert_node_with_relations(node)
        nodes.append(node)
        idx += 1

    return controller, nodes


def run_tests():
    start = time.time()
    controller, nodes = build_memories()
    build_time = time.time() - start

    logger.info("Built %d nodes in %.2f ms", len(nodes), build_time * 1000)

    # Retrieval tests by tag
    sim_nodes = controller.test_retrieval_by_tag("group_sim")
    diverse_nodes = controller.test_retrieval_by_tag("diverse")
    logger.info("Tag retrieval group_sim -> %d nodes", len(sim_nodes))
    logger.info("Tag retrieval diverse -> %d nodes", len(diverse_nodes))

    # Inspect edges from a sample similar node
    if sim_nodes:
        sample_id = sim_nodes[0].id
        edges = controller.graph.edges.get_edges(sample_id)
        logger.info("Edges %s: %d", sample_id, len(edges))
        sim_edges = [e for e in edges if e[1] == "similarity"]
        logger.info("Sample node %s has %d total edges (%d similarity)", sample_id, len(edges), len(sim_edges))
        logger.info("Top 5 similarity edges: %s", sorted(sim_edges, key=lambda x: x[2], reverse=True)[:5])
        # Proper neighbor lookup (edge tuple = (neighbor_id, relation_type, weight))
        if sim_edges:
            # Sort by weight descending and pick top
            top_edge = max(sim_edges, key=lambda x: x[2])
            neighbor_id = top_edge[0]
            neighbor_node = controller.graph.nodes.get(neighbor_id) 
            if neighbor_node:
                logger.info("Most similar pair: %s  <->  %s (w=%.3f)",
                            sim_nodes[0].context["thought"],
                            neighbor_node.context.get("thought"),
                            top_edge[2])

    # Performance metrics
    report = controller.report()
    logger.info("Report: %s", report)

    # Basic assertions / expectations (manual check logs)
    if len(sim_nodes) < 5:
        logger.warning("Expected at least 5 similar-group nodes")

    return report


if __name__ == "__main__":
    run_tests()
