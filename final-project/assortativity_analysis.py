import networkx as nx

def analyze_assortativity(G, period_label):
    """
    Calculate degree and party assortativity for a political network.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph with 'party' node attribute and 'weight' edge attribute
    period_label : str
        Label for printing (e.g., "Period 66")
    
    Returns:
    --------
    dict with assortativity measures
    """
    results = {}
    
    # Degree assortativity (unweighted)
    degree_assort = nx.degree_assortativity_coefficient(G)
    results['degree_assortativity'] = degree_assort
    
    # Degree assortativity (weighted)
    # This uses strength (sum of edge weights) rather than degree
    degree_assort_weighted = nx.degree_pearson_correlation_coefficient(G, weight='weight')
    results['degree_assortativity_weighted'] = degree_assort_weighted
    
    # Party assortativity
    # Check that all nodes have party attribute
    nodes_with_party = [n for n in G.nodes() if G.nodes[n].get('party') is not None]
    if len(nodes_with_party) < len(G.nodes()):
        print(f"Warning: {len(G.nodes()) - len(nodes_with_party)} nodes missing party attribute")
    
    party_assort = nx.attribute_assortativity_coefficient(G, 'party')
    results['party_assortativity'] = party_assort
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ASSORTATIVITY ANALYSIS: {period_label}")
    print(f"{'='*60}")
    print(f"Degree assortativity (unweighted): {degree_assort:.4f}")
    print(f"Degree assortativity (weighted):   {degree_assort_weighted:.4f}")
    print(f"Party assortativity:               {party_assort:.4f}")

    
    return results
