"""
db.py
Local Graph Database and Vector Index for Blender Documentation.
Built with SQLite (FTS5 + Graph relational schema + Vector cosine similarity).
Fully offline, zero daemon requirements, stored in ./db/blender_graph.db.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BlenderGraphDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = base_dir / "db" / "blender_graph.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize tables for nodes, edges, FTS5 full-text index, and embeddings."""
        with self.conn:
            # 1. Graph Nodes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT,
                    summary TEXT,
                    mode TEXT,
                    menu_path TEXT,
                    shortcut TEXT,
                    python_op TEXT,
                    file_path TEXT,
                    raw_content TEXT
                )
            """)

            # 2. Graph Edges
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    properties TEXT,
                    FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)

            # 3. Indices on graph
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_mode ON nodes(mode)")

            # 4. FTS5 Virtual Table for full-text search
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
                    id UNINDEXED,
                    name,
                    summary,
                    category,
                    mode,
                    menu_path,
                    shortcut,
                    raw_content,
                    tokenize='porter unicode61'
                )
            """)

            # 5. Embeddings table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS node_embeddings (
                    node_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)

    def insert_nodes_batch(self, nodes: List[Dict[str, Any]]):
        """Batch insert or replace nodes and update FTS5 in a single transaction."""
        node_tuples = []
        fts_tuples = []
        for node in nodes:
            node_tuples.append((
                node["id"],
                node.get("type", "concept"),
                node["name"],
                node.get("category", ""),
                node.get("summary", ""),
                node.get("mode"),
                node.get("menu_path"),
                node.get("shortcut"),
                node.get("python_op"),
                node.get("file_path"),
                node.get("raw_content", "")
            ))
            fts_tuples.append((
                node["id"],
                node["name"],
                node.get("summary", ""),
                node.get("category", ""),
                node.get("mode") or "",
                node.get("menu_path") or "",
                node.get("shortcut") or "",
                node.get("raw_content", "")
            ))

        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO nodes (
                    id, type, name, category, summary, mode, menu_path, shortcut, python_op, file_path, raw_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, node_tuples)

            self.conn.executemany("""
                INSERT INTO fts_nodes (
                    id, name, summary, category, mode, menu_path, shortcut, raw_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, fts_tuples)

    def insert_edges_batch(self, edges: List[Tuple[str, str, str, Optional[Dict]]]):
        """Batch insert edges in a single transaction."""
        edge_tuples = [
            (src, tgt, rel, json.dumps(props or {}))
            for src, tgt, rel, props in edges
        ]
        with self.conn:
            self.conn.executemany("""
                INSERT INTO edges (source_id, target_id, relation, properties)
                VALUES (?, ?, ?, ?)
            """, edge_tuples)

    def insert_embeddings_batch(self, items: List[Tuple[str, np.ndarray]]):
        """Batch insert embeddings in a single transaction."""
        tuples = [
            (nid, np.asarray(emb, dtype=np.float32).tobytes(), len(emb))
            for nid, emb in items
        ]
        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimension)
                VALUES (?, ?, ?)
            """, tuples)

    def insert_node(self, node: Dict[str, Any]):
        """Insert or replace a node in nodes and fts_nodes."""
        self.insert_nodes_batch([node])

    def insert_edge(self, source_id: str, target_id: str, relation: str, properties: Optional[Dict] = None):
        """Insert an edge between two nodes."""
        self.insert_edges_batch([(source_id, target_id, relation, properties)])

    def insert_embedding(self, node_id: str, embedding: np.ndarray):
        """Store dense vector embedding as a float32 binary blob."""
        self.insert_embeddings_batch([(node_id, embedding)])

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_edges_from(self, source_id: str) -> List[Dict[str, Any]]:
        """Get outgoing edges from source_id."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges WHERE source_id = ?", (source_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_edges_to(self, target_id: str) -> List[Dict[str, Any]]:
        """Get incoming edges to target_id."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges WHERE target_id = ?", (target_id,))
        return [dict(r) for r in cur.fetchall()]

    def search_fts(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search nodes using SQLite FTS5 BM25 ranking."""
        cur = self.conn.cursor()
        # Clean query for FTS5 syntax
        clean_terms = [f'"{w}"*' for w in query.split() if w.isalnum()]
        if not clean_terms:
            return []
        fts_query = " OR ".join(clean_terms)
        try:
            cur.execute("""
                SELECT n.*, bm25(fts_nodes) as fts_rank
                FROM fts_nodes
                JOIN nodes n ON fts_nodes.id = n.id
                WHERE fts_nodes MATCH ?
                ORDER BY fts_rank ASC
                LIMIT ?
            """, (fts_query, limit))
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            # Fallback to simple LIKE
            pattern = f"%{query}%"
            cur.execute("""
                SELECT * FROM nodes
                WHERE name LIKE ? OR summary LIKE ? OR menu_path LIKE ?
                LIMIT ?
            """, (pattern, pattern, pattern, limit))
            return [dict(r) for r in cur.fetchall()]

    def search_vector(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        """Search nodes using cosine similarity over precomputed embeddings."""
        cur = self.conn.cursor()
        cur.execute("SELECT node_id, embedding, dimension FROM node_embeddings")
        rows = cur.fetchall()

        if not rows:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        node_ids = []
        vectors = []
        for r in rows:
            node_ids.append(r["node_id"])
            vectors.append(np.frombuffer(r["embedding"], dtype=np.float32))

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-10

        similarities = np.dot(matrix, q_vec) / (norms * q_norm)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            nid = node_ids[idx]
            score = float(similarities[idx])
            node = self.get_node(nid)
            if node:
                results.append((node, score))

        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        alpha: float = 0.65
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense Vector Cosine Similarity and Sparse FTS5 BM25.
        alpha: weight for vector score (1.0 = pure vector, 0.0 = pure FTS).
        """
        scores: Dict[str, float] = {}
        nodes: Dict[str, Dict[str, Any]] = {}

        # 1. Vector Search
        if query_embedding is not None:
            vec_results = self.search_vector(query_embedding, top_k=top_k * 2)
            for node, sim in vec_results:
                nid = node["id"]
                nodes[nid] = node
                # Normalized similarity [-1, 1] -> [0, 1]
                scores[nid] = scores.get(nid, 0.0) + alpha * max(0.0, sim)

        # 2. FTS Search
        fts_results = self.search_fts(query, limit=top_k * 2)
        if fts_results:
            # BM25 ranks are negative (closer to 0 is better or negative)
            min_rank = min(r.get("fts_rank", 0) for r in fts_results)
            max_rank = max(r.get("fts_rank", 0) for r in fts_results)
            span = max(abs(max_rank - min_rank), 1e-5)

            for r in fts_results:
                nid = r["id"]
                nodes[nid] = r
                rank = r.get("fts_rank", 0)
                # Invert BM25 so best match is closest to 1.0
                norm_score = 1.0 - (rank - min_rank) / span
                scores[nid] = scores.get(nid, 0.0) + (1.0 - alpha) * max(0.0, norm_score)

        # Sort combined results
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        return [{**nodes[nid], "score": round(scores[nid], 4)} for nid in sorted_ids]

    def get_procedural_context(self, node_id: str) -> Dict[str, Any]:
        """
        Traverse graph to resolve procedural workflow steps for a target node:
        Finds:
        - Parent Editor
        - Active Mode
        - Menu Path or Panel Tab
        - Shortcut / Keymap
        - Related parameters / tools
        """
        node = self.get_node(node_id)
        if not node:
            return {}

        cur = self.conn.cursor()

        # Find connected editors
        cur.execute("""
            SELECT n.* FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id = ? AND e.relation = 'IN_EDITOR'
            UNION
            SELECT n.* FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id = ? AND e.relation = 'IN_EDITOR'
        """, (node_id, node_id))
        editors = [dict(r) for r in cur.fetchall()]

        # Find required modes
        cur.execute("""
            SELECT n.* FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id = ? AND e.relation = 'REQUIRES_MODE'
        """, (node_id,))
        modes = [dict(r) for r in cur.fetchall()]

        # Find shortcuts
        cur.execute("""
            SELECT n.* FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id = ? AND e.relation = 'TRIGGERED_BY'
        """, (node_id,))
        shortcuts = [dict(r) for r in cur.fetchall()]

        # Find related operators/tools
        cur.execute("""
            SELECT n.*, e.relation FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id = ? AND e.relation IN ('RELATED_TO', 'HAS_SETTING')
            LIMIT 5
        """, (node_id,))
        related = [dict(r) for r in cur.fetchall()]

        return {
            "node": node,
            "editors": editors,
            "modes": modes,
            "shortcuts": shortcuts,
            "related": related
        }
