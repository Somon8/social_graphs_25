import pandas as pd
import networkx as nx
import numpy as np


def calculate_centrality_measures(G, network_name="Network"):
    """
    Calculate multiple centrality measures for a network.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze
    network_name : str
        Name for printing (e.g., "Full Network", "Backbone Network")
    
    Returns:
    --------
    dict : Dictionary with centrality measure names as keys and dicts of {node: score} as values
    """
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
    print("  - Computing degree centrality...")
    centrality_measures['degree'] = nx.degree_centrality(G)
    
    # Betweenness Centrality
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
    print("  - Computing closeness centrality...")
    if has_weights:
        centrality_measures['closeness'] = nx.closeness_centrality(G_copy, distance='distance')
    else:
        centrality_measures['closeness'] = nx.closeness_centrality(G)
    
    # Eigenvector Centrality
    print("  - Computing eigenvector centrality...")
    try:
        if has_weights:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        else:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("    WARNING: Eigenvector centrality failed to converge, skipping...")
        centrality_measures['eigenvector'] = None
    except Exception as e:
        print(f"    WARNING: Eigenvector centrality failed ({str(e)}), skipping...")
        centrality_measures['eigenvector'] = None
    
    print(f"✓ Centrality calculations complete for {network_name}\n")
    
    return centrality_measures


def print_top_nodes(centrality_dict, measure_name, top_n=10, node_party_map=None):
    """
    Print top N nodes for a centrality measure.
    
    Parameters:
    -----------
    centrality_dict : dict
        {node: score} dictionary
    measure_name : str
        Name of the measure for printing
    top_n : int
        Number of top nodes to display
    node_party_map : dict or None
        Optional mapping of {node: party} for display
    """
    if centrality_dict is None:
        return
    
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\n{'─'*80}")
    print(f"Top {top_n} by {measure_name}")
    print(f"{'─'*80}")
    print(f"{'Rank':<6} {'Politician':<40} {'Party':<25} {'Score':<10}")
    print(f"{'─'*80}")
    
    for rank, (node, score) in enumerate(sorted_nodes, 1):
        party = node_party_map.get(node, 'Unknown') if node_party_map else 'N/A'
        # Truncate long names
        node_display = node[:37] + '...' if len(node) > 40 else node
        party_display = party[:22] + '...' if len(party) > 25 else party
        print(f"{rank:<6} {node_display:<40} {party_display:<25} {score:.4f}")


def compare_centrality_measures(centrality_full, centrality_backbone, 
                                measure_name, top_n=10, node_party_map=None):
    """
    Compare centrality rankings between full and backbone networks.
    
    Parameters:
    -----------
    centrality_full : dict
        Centrality scores from full network
    centrality_backbone : dict
        Centrality scores from backbone network
    measure_name : str
        Name of the centrality measure
    top_n : int
        Number of top changes to show
    node_party_map : dict or None
        Optional mapping of {node: party}
    """
    if centrality_full is None or centrality_backbone is None:
        return
    
    # Find nodes present in both
    common_nodes = set(centrality_full.keys()) & set(centrality_backbone.keys())
    
    # Calculate rank changes
    rank_full = {node: rank for rank, (node, _) in enumerate(
        sorted(centrality_full.items(), key=lambda x: x[1], reverse=True), 1)}
    rank_backbone = {node: rank for rank, (node, _) in enumerate(
        sorted(centrality_backbone.items(), key=lambda x: x[1], reverse=True), 1)}
    
    rank_changes = []
    for node in common_nodes:
        change = rank_full[node] - rank_backbone[node]  # Positive = improved in backbone
        rank_changes.append((node, change, rank_full[node], rank_backbone[node],
                           centrality_full[node], centrality_backbone[node]))
    
    # Biggest improvers
    biggest_improvers = sorted(rank_changes, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Biggest decliners
    biggest_decliners = sorted(rank_changes, key=lambda x: x[1])[:top_n]
    
    print(f"\n{'='*80}")
    print(f"RANK CHANGES: {measure_name}")
    print(f"{'='*80}")
    
    # Improvers
    print(f"\n{'─'*80}")
    print(f"Top {top_n} Biggest Improvers (More Important in Backbone)")
    print(f"{'─'*80}")
    print(f"{'Politician':<40} {'Party':<20} {'Full→BB':<12} {'Score Change':<15}")
    print(f"{'─'*80}")
    
    for node, change, rank_f, rank_b, score_f, score_b in biggest_improvers:
        if change <= 0:
            continue
        party = node_party_map.get(node, 'Unknown') if node_party_map else 'N/A'
        node_display = node[:37] + '...' if len(node) > 40 else node
        party_display = party[:17] + '...' if len(party) > 20 else party
        rank_str = f"#{rank_f}→#{rank_b}"
        score_change = f"{score_f:.4f}→{score_b:.4f}"
        print(f"{node_display:<40} {party_display:<20} {rank_str:<12} {score_change:<15}")
    
    # Decliners
    print(f"\n{'─'*80}")
    print(f"Top {top_n} Biggest Decliners (Less Important in Backbone)")
    print(f"{'─'*80}")
    print(f"{'Politician':<40} {'Party':<20} {'Full→BB':<12} {'Score Change':<15}")
    print(f"{'─'*80}")
    
    for node, change, rank_f, rank_b, score_f, score_b in biggest_decliners:
        if change >= 0:
            continue
        party = node_party_map.get(node, 'Unknown') if node_party_map else 'N/A'
        node_display = node[:37] + '...' if len(node) > 40 else node
        party_display = party[:17] + '...' if len(party) > 20 else party
        rank_str = f"#{rank_f}→#{rank_b}"
        score_change = f"{score_f:.4f}→{score_b:.4f}"
        print(f"{node_display:<40} {party_display:<20} {rank_str:<12} {score_change:<15}")


def full_centrality_analysis(df_agg, G_full, G_backbone, print_stats=True):
    """
    Run complete centrality analysis comparing full and backbone networks.
    
    Parameters:
    -----------
    df_agg : pandas.DataFrame
        Aggregated edge list (full network)
    G_full : networkx.Graph
        Full network graph
    G_backbone : networkx.Graph
        Backbone network graph
    
    Returns:
    --------
    tuple : (centrality_full_dict, centrality_backbone_dict)
    """
    # Create node to party mapping
    node_party_map = {}
    for _, row in df_agg.iterrows():
        node_party_map[row['source']] = row['source_party']
        node_party_map[row['target']] = row['target_party']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CENTRALITY ANALYSIS")
    print("="*80)
    
    # Calculate centrality for both networks
    centrality_full = calculate_centrality_measures(G_full, "Full Network")
    centrality_backbone = calculate_centrality_measures(G_backbone, "Backbone Network")
    
    # Convert to DataFrames with politicians as rows and measures as columns
    df_centrality_full = pd.DataFrame({
        measure: scores for measure, scores in centrality_full.items() if scores is not None
    })
    df_centrality_backbone = pd.DataFrame({
        measure: scores for measure, scores in centrality_backbone.items() if scores is not None
    })
    if print_stats == True:
        # Print top nodes for each measure in each network
        print("\n" + "="*80)
        print("FULL NETWORK - TOP POLITICIANS BY CENTRALITY")
        print("="*80)
        
        for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
            if centrality_full.get(measure):
                print_top_nodes(centrality_full[measure], 
                            f"{measure.capitalize()} Centrality (Full Network)", 
                            top_n=10, 
                            node_party_map=node_party_map)
        
        print("\n" + "="*80)
        print("BACKBONE NETWORK - TOP POLITICIANS BY CENTRALITY")
        print("="*80)
        
        for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
            if centrality_backbone.get(measure):
                print_top_nodes(centrality_backbone[measure], 
                            f"{measure.capitalize()} Centrality (Backbone)", 
                            top_n=10, 
                            node_party_map=node_party_map)
        
        # Compare rankings
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS: FULL vs BACKBONE")
        print("="*80)
        
        for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
            if centrality_full.get(measure) and centrality_backbone.get(measure):
                compare_centrality_measures(centrality_full[measure],
                                        centrality_backbone[measure],
                                        f"{measure.capitalize()} Centrality",
                                        top_n=10,
                                        node_party_map=node_party_map)
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
            if centrality_full.get(measure) and centrality_backbone.get(measure):
                scores_full = list(centrality_full[measure].values())
                scores_backbone = list(centrality_backbone[measure].values())
                
                print(f"\n{measure.capitalize()} Centrality:")
                print(f"  Full Network    - Mean: {np.mean(scores_full):.4f}, Std: {np.std(scores_full):.4f}")
                print(f"  Backbone Network - Mean: {np.mean(scores_backbone):.4f}, Std: {np.std(scores_backbone):.4f}")
        
        print("\n" + "="*80 + "\n")
        
    return df_centrality_full, df_centrality_backbone


# Example usage:
"""
from political_network_backboning import filter_and_aggregate, apply_backboning

# Step 1: Aggregate data
df_agg = filter_and_aggregate(df, periods=[65, 66, 67])

# Step 2: Build full network
G_full = nx.from_pandas_edgelist(df_agg, 'source', 'target', edge_attr='weight')

# Step 3: Apply backboning
df_backbone, stats, G_backbone = apply_backboning(df_agg, min_votes_threshold=10, alpha=0.1)

# Step 4: Run comprehensive centrality analysis
centrality_full, centrality_backbone = full_centrality_analysis(
    df_agg, df_backbone, G_full, G_backbone
)
"""