import pandas as pd
import networkx as nx
import numpy as np


def centrality_analysis(G, network_name="Network", print_stats=True):
    """
    Calculate multiple centrality measures for a network.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
    network_name : str
        Name for printing (e.g., "Full Network", "Period 66")
    print_stats : bool
        Whether to print progress messages
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with politicians as index and centrality measures as columns
    """
    if print_stats:
        print(f"\n{'='*80}")
        print(f"Calculating centrality measures for {network_name}")
        print(f"{'='*80}")
        print(f"Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    centrality_measures = {}
    
    # Check if graph has weights
    has_weights = False
    for u, v, d in G.edges(data=True):
        if 'weight' in d:
            has_weights = True
            break
    
    # Degree Centrality
    if print_stats:
        print("  - Computing degree centrality...")
    centrality_measures['degree'] = nx.degree_centrality(G)
    
    # Betweenness Centrality
    if print_stats:
        print("  - Computing betweenness centrality...")
    if has_weights:
        # For betweenness, higher weight = shorter distance, so we invert
        # Create a copy to avoid modifying original graph
        G_copy = G.copy()
        for u, v, d in G_copy.edges(data=True):
            weight = d.get('weight', 1.0)
            d['distance'] = 1.0 / (weight + 1e-10)  # Avoid division by zero
        centrality_measures['betweenness'] = nx.betweenness_centrality(G_copy, weight='distance')
    else:
        # Unweighted betweenness
        centrality_measures['betweenness'] = nx.betweenness_centrality(G)
    
    # Closeness Centrality
    if print_stats:
        print("  - Computing closeness centrality...")
    if has_weights:
        centrality_measures['closeness'] = nx.closeness_centrality(G_copy, distance='distance')
    else:
        centrality_measures['closeness'] = nx.closeness_centrality(G)
    
    # Eigenvector Centrality
    if print_stats:
        print("  - Computing eigenvector centrality...")
    try:
        if has_weights:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        else:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        if print_stats:
            print("    WARNING: Eigenvector centrality failed to converge, skipping...")
        centrality_measures['eigenvector'] = None
    except Exception as e:
        if print_stats:
            print(f"    WARNING: Eigenvector centrality failed ({str(e)}), skipping...")
        centrality_measures['eigenvector'] = None
    
    if print_stats:
        print(f"✓ Centrality calculations complete for {network_name}\n")
    
    # Convert to DataFrame with politicians as rows and measures as columns
    df_centrality = pd.DataFrame({
        measure: scores for measure, scores in centrality_measures.items() if scores is not None
    })
    
    return df_centrality
