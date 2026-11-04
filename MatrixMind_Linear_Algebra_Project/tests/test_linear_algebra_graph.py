from src.knowledge_graph.graph_store import extract_concepts


def test_linear_algebra_concepts_are_extracted():
    found = extract_concepts('Gaussian elimination gives row echelon form; eigenvalues and eigenvectors are important.')
    assert 'gaussian elimination' in found
    assert 'row echelon form' in found
    assert 'eigenvalue' in found
    assert 'eigenvector' in found


def test_ontology_relationships_exist():
    from src.knowledge_graph.graph_store import ONTOLOGY_RELATIONS
    assert ("gaussian elimination", "row echelon form", "PRODUCES") in ONTOLOGY_RELATIONS
    assert ("eigenvalue", "eigenvector", "PAIRED_WITH") in ONTOLOGY_RELATIONS


def test_rag_retrieval_expands_related_concepts(tmp_path):
    import networkx as nx
    from src.embeddings.embed_documents import EmbeddingEngine
    from src.knowledge_graph.graph_store import KnowledgeGraphStore
    from src.retrieval.retriever import Retriever

    engine = EmbeddingEngine()
    store = KnowledgeGraphStore(path=tmp_path / 'kg.json', embedding_engine=engine)
    graph = nx.Graph()
    text_a = 'Characteristic polynomial helps determine the spectrum of a square matrix.'
    text_b = 'Eigenvector is a non-zero vector that is scaled by its corresponding eigenvalue.'
    graph.add_node('chunk-a', node_type='chunk', source='notes.txt', text=text_a,
                   embedding=engine.embed_query('characteristic polynomial').tolist(), chunk_id=0)
    graph.add_node('concept-a', node_type='concept', label='characteristic polynomial')
    graph.add_node('concept-b', node_type='concept', label='eigenvector')
    graph.add_node('chunk-b', node_type='chunk', source='notes.txt', text=text_b,
                   embedding=[0.0] * engine.get_embedding_dim(), chunk_id=1)
    graph.add_edge('chunk-a', 'concept-a', relation='MENTIONS')
    graph.add_edge('concept-a', 'concept-b', relation='RELATED_TO')
    graph.add_edge('concept-b', 'chunk-b', relation='MENTIONS')
    store.graph = graph
    store.loaded = True

    hits = Retriever(graph_store=store).retrieve('characteristic polynomial', top_k=5, threshold=0.99, use_graph=True)
    assert any(hit.text == text_a for hit in hits)
    assert any(hit.text == text_b for hit in hits)
