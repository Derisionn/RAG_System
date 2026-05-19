from neo4j import GraphDatabase
from ..config.config import NEO4J_URI, NEO4J_USER, NEO4J_PWD

class GraphService:
    def __init__(self):
        self.driver = self._make_driver()

    def _make_driver(self) -> GraphDatabase.driver:
        """Create a resilient connection driver to Neo4j."""
        return GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PWD),
            max_connection_lifetime=200,
            max_connection_pool_size=5,
            connection_acquisition_timeout=30,
        )

    def find_join_paths(self, tables: list[str]) -> list[list[str]]:
        """Resiliently query Cypher for shortest join paths with safety retry and graceful fallback."""
        cypher = """
        MATCH (a:Table {name: $start_node}), (b:Table {name: $end_node})
        MATCH p = shortestPath((a)-[:REFERENCES*..3]-(b))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """
        paths: list[list[str]] = []
        if len(tables) < 2:
            return paths

        try:
            paths = self._execute_find_paths(cypher, tables)
        except Exception as e:
            print(f"  [WARNING] Neo4j session error ({type(e).__name__}): {e}. Recreating driver and retrying...")
            try:
                self.driver.close()
                self.driver = self._make_driver()
                paths = self._execute_find_paths(cypher, tables)
            except Exception as retry_err:
                print(f"  [ERROR] Graceful degradation: Neo4j retry failed: {retry_err}. Proceeding without join paths.")
                paths = []
        return paths

    def _execute_find_paths(self, cypher: str, tables: list[str]) -> list[list[str]]:
        paths = []
        with self.driver.session() as session:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    result = session.run(cypher, start_node=tables[i], end_node=tables[j])
                    for record in result:
                        path = record["path_nodes"]
                        if path and len(path) > 1:
                            paths.append(path)
        return paths

    def count_nodes(self) -> int:
        """Get table count from Neo4j for health check purposes."""
        with self.driver.session() as session:
            return session.run("MATCH (t:Table) RETURN count(t) AS n").single()["n"]

    def close(self):
        """Close database driver resources."""
        self.driver.close()
