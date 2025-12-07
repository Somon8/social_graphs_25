import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import sys
import os
import urllib.request


# ============================================================================
# ASSORTATIVITY
# ============================================================================

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
    nodes_with_party = [n for n in G.nodes() if G.nodes[n].get('party') is not None]
    if len(nodes_with_party) < len(G.nodes()):
        print(f"Warning: {len(G.nodes()) - len(nodes_with_party)} nodes missing party attribute")
    
    party_assort = nx.attribute_assortativity_coefficient(G, 'party')
    results['party_assortativity'] = party_assort
    
    # Print summary
    print(f"\nASSORTATIVITY ANALYSIS: {period_label}")
    print(f"{'='*60}")
    print(f"Degree assortativity (unweighted): {degree_assort:.4f}")
    print(f"Degree assortativity (weighted):   {degree_assort_weighted:.4f}")
    print(f"Party assortativity:               {party_assort:.4f}")
    
    return results


# ============================================================================
# DATA LOADING
# ============================================================================

def from_github(path):
    base_url = "https://raw.githubusercontent.com/Somon8/social_graphs_25/main/final-project"
    url_to_file = base_url + path
    print(f"Fetching from GitHub: {url_to_file}")
    return urllib.request.urlopen(url_to_file)


# ============================================================================
# HIGH SALIENCE SKELETON / BACKBONE
# ============================================================================

def high_salience_skeleton(table, undirected=False, return_self_loops=False):
    """
    Calculate high salience skeleton backbone (from course materials).
    
    This function identifies statistically significant edges based on shortest path
    calculations through the network.
    """
    sys.stderr.write("Calculating HSS score...\n")
    table = table.copy()
    table['distance'] = 1.0 / table['nij']
    nodes = set(table['src']) | set(table['trg'])
    G = nx.from_pandas_edgelist(table, source='src', target='trg', 
                                 edge_attr='distance', create_using=nx.DiGraph())
    cs = defaultdict(float)
    
    for s in nodes:
        pred = defaultdict(list)
        dist = {t: float('inf') for t in nodes}
        dist[s] = 0.0
        Q = defaultdict(list)
        for w in dist:
            Q[dist[w]].append(w)
        S = []
        
        while len(Q) > 0:
            v = Q[min(Q.keys())].pop(0)
            S.append(v)
            for _, w, l in G.edges(nbunch=[v], data=True):
                new_distance = dist[v] + l['distance']
                if dist[w] > new_distance:
                    Q[dist[w]].remove(w)
                    dist[w] = new_distance
                    Q[dist[w]].append(w)
                    pred[w] = []
                if dist[w] == new_distance:
                    pred[w].append(v)
            while len(S) > 0:
                w = S.pop()
                for v in pred[w]:
                    cs[(v, w)] += 1.0
            Q = defaultdict(list, {k: v for k, v in Q.items() if len(v) > 0})
    
    table['score'] = table.apply(lambda x: cs[(x['src'], x['trg'])] / len(nodes), axis=1)
    
    if not return_self_loops:
        table = table[table['src'] != table['trg']]
    
    if undirected:
        table['edge'] = table.apply(lambda x: '%s-%s' % (min(x['src'], x['trg']), 
                                                          max(x['src'], x['trg'])), axis=1)
        table_maxscore = table.groupby(by='edge')['score'].sum().reset_index()
        table = table.merge(table_maxscore, on='edge', suffixes=('_min', ''))
        table = table.drop_duplicates(subset=['edge'])
        table = table.drop('edge', axis=1)
        table = table.drop('score_min', axis=1)
        table['score'] = table['score'] / 2.0
    
    return table[['src', 'trg', 'nij', 'score']]


def apply_backboning(df, 
                     source_col='source',
                     target_col='target', 
                     weight_col='weight',
                     min_votes_threshold=None,
                     votes_col='total_votes_shared',
                     alpha=0.0):
    """
    Apply high_salience_skeleton backboning to political voting network.
    """
    df = df.copy()
    
    # Store original statistics
    original_edges = len(df)
    original_nodes = len(set(df[source_col]) | set(df[target_col]))
    
    # Filter by minimum votes threshold BEFORE backboning
    if min_votes_threshold is not None:
        df = df[df[votes_col] >= min_votes_threshold]
        after_filter_edges = len(df)
        after_filter_nodes = len(set(df[source_col]) | set(df[target_col]))
    else:
        after_filter_edges = original_edges
        after_filter_nodes = original_nodes
    
    # Prepare edge table for backboning
    edge_table = df[[source_col, target_col, weight_col]].copy()
    edge_table.columns = ['src', 'trg', 'nij']
    
    # Apply high salience skeleton
    backbone_table = high_salience_skeleton(edge_table, undirected=True)
    
    # Apply alpha threshold
    backbone_table = backbone_table[backbone_table['score'] > alpha]
    
    # Merge back with original data to get ALL original columns
    backbone_table = backbone_table.merge(
        df, 
        left_on=['src', 'trg'], 
        right_on=[source_col, target_col],
        how='left'
    )
    
    # Clean up: keep original columns plus hss_score
    cols_to_keep = [source_col, target_col, 'source_party', 'target_party', 
                    votes_col, 'total_votes_agreed', weight_col]
    
    backbone_table['hss_score'] = backbone_table['score']
    cols_to_keep.append('hss_score')
    
    result_df = backbone_table[cols_to_keep].copy()
    
    # Create NetworkX graph with all attributes
    G = nx.from_pandas_edgelist(
        result_df, 
        source=source_col, 
        target=target_col,
        edge_attr=[weight_col, 'hss_score', votes_col, 'total_votes_agreed'],
        create_using=nx.Graph()
    )
    
    # Add party information as node attributes
    node_parties = {}
    for _, row in result_df.iterrows():
        node_parties[row[source_col]] = row['source_party']
        node_parties[row[target_col]] = row['target_party']
    
    nx.set_node_attributes(G, node_parties, 'party')
    
    # Calculate statistics
    backbone_edges = len(result_df)
    backbone_nodes = G.number_of_nodes()
    
    stats = {
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'after_min_votes_filter_nodes': after_filter_nodes,
        'after_min_votes_filter_edges': after_filter_edges,
        'backbone_nodes': backbone_nodes,
        'backbone_edges': backbone_edges,
        'nodes_retained_pct': 100.0 * backbone_nodes / original_nodes,
        'edges_retained_pct': 100.0 * backbone_edges / original_edges,
        'min_votes_threshold': min_votes_threshold,
        'alpha_threshold': alpha,
        'avg_weight': result_df[weight_col].mean(),
        'avg_hss_score': result_df['hss_score'].mean(),
    }
    
    # Print summary
    print("\n" + "="*80)
    print("BACKBONING SUMMARY")
    print("="*80)
    print(f"\nOriginal network:")
    print(f"  Nodes: {original_nodes}")
    print(f"  Edges: {original_edges}")
    
    if min_votes_threshold is not None:
        print(f"\nAfter min_votes_threshold={min_votes_threshold}:")
        print(f"  Nodes: {after_filter_nodes} ({100.0 * after_filter_nodes / original_nodes:.1f}%)")
        print(f"  Edges: {after_filter_edges} ({100.0 * after_filter_edges / original_edges:.1f}%)")
    
    print(f"\nAfter backbone extraction (alpha={alpha}):")
    print(f"  Nodes: {backbone_nodes} ({stats['nodes_retained_pct']:.1f}%)")
    print(f"  Edges: {backbone_edges} ({stats['edges_retained_pct']:.1f}%)")
    print(f"\nBackbone edge statistics:")
    print(f"  Average weight (agreement rate): {stats['avg_weight']:.3f}")
    print(f"  Average HSS score: {stats['avg_hss_score']:.4f}")
    print(f"  Weight range: [{result_df[weight_col].min():.3f}, {result_df[weight_col].max():.3f}]")
    print(f"  HSS score range: [{result_df['hss_score'].min():.4f}, {result_df['hss_score'].max():.4f}]")
    print("="*80 + "\n")
    
    return result_df, stats, G


# ============================================================================
# MODULARITY
# ============================================================================

def calculate_modularity(G, communities, verbose=False):
    L = G.number_of_edges()
    M = 0
    
    for community_label, nodes in communities.items():
        Lc = 0  # Internal edges in community
        kc = 0  # Sum of degrees in community     
        for node1 in nodes:
            if node1 in G:  # Check if node exists in graph
                kc += G.degree(node1)
                for node2 in nodes:
                    if G.has_edge(node1, node2):
                        Lc += 0.5  # Count each edge once      
        left = Lc / L
        right = (kc / (2 * L)) ** 2
        Mc = left - right
        
        if verbose:
            print(f"Community '{community_label}':")
            print(f"  Internal edges (Lc): {Lc}")
            print(f"  Total degree (kc): {kc}")
            print(f"  Modularity contribution (Mc): {Mc:.4f}")
        
        M += Mc
    
    return M


# ============================================================================
# CENTRALITY
# ============================================================================

def centrality_analysis(G, network_name="Network", print_stats=True):
    """
    Calculate multiple centrality measures for a network.
    
    Returns a pandas.DataFrame with nodes as index and
    columns: 'degree', 'betweenness', 'closeness', 'eigenvector'.
    """
    if print_stats:
        print(f"\n{'='*80}")
        print(f"Calculating centrality measures for {network_name}")
        print(f"{'='*80}")
        print(f"Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    centrality_measures = {}
    
    # Check if graph has weights
    has_weights = any('weight' in d for _, _, d in G.edges(data=True))
    
    # Degree Centrality
    if print_stats:
        print("  - Computing degree centrality...")
    centrality_measures['degree'] = nx.degree_centrality(G)
    
    # Betweenness Centrality
    if print_stats:
        print("  - Computing betweenness centrality...")
    if has_weights:
        G_copy = G.copy()
        for u, v, d in G_copy.edges(data=True):
            weight = d.get('weight', 1.0)
            d['distance'] = 1.0 / (weight + 1e-10)
        centrality_measures['betweenness'] = nx.betweenness_centrality(G_copy, weight='distance')
    else:
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
    
    df_centrality = pd.DataFrame({
        measure: scores for measure, scores in centrality_measures.items() if scores is not None
    })
    
    return df_centrality


# ============================================================================
# WRAPPER FUNCTIONS FOR STATS
# ============================================================================

def summarize_distribution(values, prefix):
    """Compute summary statistics for a list of values."""
    if len(values) == 0:
        return {f'{prefix}_{s}': np.nan for s in 
                ['mean', 'median', 'max', 'min', 'std', 'variance', 'skewness', 'kurtosis']}
    
    arr = np.array(values)
    stats = {
        f'{prefix}_mean': np.mean(arr),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_max': np.max(arr),
        f'{prefix}_min': np.min(arr),
        f'{prefix}_std': np.std(arr),
        f'{prefix}_variance': np.var(arr),
    }
    
    if np.std(arr) > 0:
        standardized = (arr - np.mean(arr)) / np.std(arr)
        stats[f'{prefix}_skewness'] = np.mean(standardized ** 3)
        stats[f'{prefix}_kurtosis'] = np.mean(standardized ** 4) - 3
    else:
        stats[f'{prefix}_skewness'] = 0
        stats[f'{prefix}_kurtosis'] = 0
    
    return stats


def extract_centrality_stats(centrality_df):
    """
    Extract summary statistics from the centrality DataFrame returned by centrality_analysis().
    """
    stats = {}
    measures = ['degree', 'betweenness', 'closeness', 'eigenvector']
    suffixes = ['mean', 'median', 'max', 'min', 'std', 'variance', 'skewness', 'kurtosis']
    
    for measure in measures:
        if measure in centrality_df.columns:
            values = centrality_df[measure].dropna().values
            if len(values) > 0:
                measure_stats = summarize_distribution(values, measure)
                stats.update(measure_stats)
                continue
        
        # If column missing or empty
        for s in suffixes:
            stats[f'{measure}_{s}'] = np.nan
    
    return stats


def extract_assortativity_stats(G, period_label=""):
    """
    Wrapper around analyze_assortativity() that suppresses printing.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        results = analyze_assortativity(G, period_label)
    except Exception:
        results = {
            'degree_assortativity': np.nan,
            'degree_assortativity_weighted': np.nan,
            'party_assortativity': np.nan
        }
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    return results


def extract_centrality_stats_silent(G, network_name="Network"):
    """
    Wrapper around centrality_analysis() that suppresses printing and returns summary stats.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    measures = ['degree', 'betweenness', 'closeness', 'eigenvector']
    suffixes = ['mean', 'median', 'max', 'min', 'std', 'variance', 'skewness', 'kurtosis']
    
    try:
        centrality_df = centrality_analysis(G, network_name=network_name, print_stats=False)
        stats = extract_centrality_stats(centrality_df)
    except Exception:
        stats = {}
        for measure in measures:
            for s in suffixes:
                stats[f'{measure}_{s}'] = np.nan
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    return stats


def compute_modularity_metrics(G, calculate_modularity_func):
    """
    Compute party-based and Louvain modularity using the provided calculate_modularity function.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {
            'party_modularity': np.nan, 
            'louvain_modularity': np.nan, 
            'num_louvain_communities': 0, 
            'num_parties': 0,
            'louvain_largest_community_size': 0,
            'louvain_largest_community_pct': np.nan,
            'louvain_num_parties_in_largest': 0,
            'louvain_dominant_party_in_largest': '',
            'louvain_dominant_party_pct_in_largest': np.nan,
            'louvain_community_composition': '',
        }
    
    stats = {}
    
    party_communities = defaultdict(list)
    for node, data in G.nodes(data=True):
        party = data.get('party')
        if party:
            party_communities[party].append(node)
    
    stats['party_modularity'] = calculate_modularity_func(G, dict(party_communities))
    stats['num_parties'] = len(party_communities)
    
    try:
        louvain_communities = nx.community.louvain_communities(G, seed=42)
        louvain_dict = {f'c{i}': list(c) for i, c in enumerate(louvain_communities)}
        stats['louvain_modularity'] = calculate_modularity_func(G, louvain_dict)
        stats['num_louvain_communities'] = len(louvain_communities)
        
        community_analysis = analyze_louvain_communities(G, louvain_communities)
        stats.update(community_analysis)
        
    except Exception:
        stats['louvain_modularity'] = np.nan
        stats['num_louvain_communities'] = 0
        stats['louvain_largest_community_size'] = 0
        stats['louvain_largest_community_pct'] = np.nan
        stats['louvain_num_parties_in_largest'] = 0
        stats['louvain_dominant_party_in_largest'] = ''
        stats['louvain_dominant_party_pct_in_largest'] = np.nan
        stats['louvain_community_composition'] = ''
    
    return stats


def analyze_louvain_communities(G, louvain_communities):
    """
    Analyze the party composition of Louvain-detected communities.
    """
    stats = {}
    total_nodes = G.number_of_nodes()
    
    sorted_communities = sorted(louvain_communities, key=len, reverse=True)
    
    community_summaries = []
    
    for i, community in enumerate(sorted_communities):
        party_counts = defaultdict(int)
        for node in community:
            party = G.nodes[node].get('party', 'Unknown')
            party_counts[party] += 1
        
        sorted_parties = sorted(party_counts.items(), key=lambda x: x[1], reverse=True)
        
        summary = ','.join([f"{p}:{c}" for p, c in sorted_parties[:5]])
        community_summaries.append(f"C{i}({len(community)}):[{summary}]")
        
        if i == 0:
            stats['louvain_largest_community_size'] = len(community)
            stats['louvain_largest_community_pct'] = len(community) / total_nodes * 100
            stats['louvain_num_parties_in_largest'] = len(party_counts)
            
            dominant_party, dominant_count = sorted_parties[0]
            stats['louvain_dominant_party_in_largest'] = dominant_party
            stats['louvain_dominant_party_pct_in_largest'] = dominant_count / len(community) * 100
    
    stats['louvain_community_composition'] = ' | '.join(community_summaries)
    stats['louvain_party_fragmentation'] = calculate_party_fragmentation(G, sorted_communities)
    
    return stats


def calculate_party_fragmentation(G, sorted_communities):
    """
    Calculate how fragmented parties are across Louvain communities.
    """
    party_members = defaultdict(list)
    for node, data in G.nodes(data=True):
        party = data.get('party', 'Unknown')
        party_members[party].append(node)
    
    node_to_community = {}
    for i, community in enumerate(sorted_communities):
        for node in community:
            node_to_community[node] = i
    
    fragmentation_scores = []
    
    for party, members in party_members.items():
        if len(members) < 2:
            continue
            
        community_counts = defaultdict(int)
        for member in members:
            comm = node_to_community.get(member, -1)
            community_counts[comm] += 1
        
        total = len(members)
        hhi = sum((count / total) ** 2 for count in community_counts.values())
        
        fragmentation_scores.append(1 - hhi)
    
    if fragmentation_scores:
        return np.mean(fragmentation_scores)
    return 0.0


def compute_basic_graph_stats(G):
    """Compute basic network statistics."""
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
    }
    
    if G.number_of_nodes() == 0:
        stats.update({
            'density': np.nan, 'avg_degree': np.nan, 'avg_clustering': np.nan,
            'num_components': 0, 'avg_shortest_path': np.nan, 'diameter': np.nan
        })
        return stats
    
    stats['density'] = nx.density(G)
    stats['avg_degree'] = 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    stats['avg_clustering'] = nx.average_clustering(G)
    stats['num_components'] = nx.number_connected_components(G)
    
    if nx.is_connected(G):
        stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    elif G.number_of_nodes() > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        if subG.number_of_nodes() > 1:
            stats['avg_shortest_path'] = nx.average_shortest_path_length(subG)
            stats['diameter'] = nx.diameter(subG)
        else:
            stats['avg_shortest_path'] = np.nan
            stats['diameter'] = np.nan
    else:
        stats['avg_shortest_path'] = np.nan
        stats['diameter'] = np.nan
    
    return stats


# ============================================================================
# GRAPH BUILDING
# ============================================================================

def build_graph_with_threshold(edgelist, threshold):
    """
    Build graph from edgelist with threshold filtering.
    """
    filtered = edgelist[edgelist['weight'] > threshold].copy()
    
    if len(filtered) == 0:
        return nx.Graph()
    
    G = nx.from_pandas_edgelist(filtered, 'source', 'target', edge_attr='weight')
    
    for _, row in filtered.iterrows():
        if row['source'] in G.nodes():
            G.nodes[row['source']]['party'] = row['source_party']
        if row['target'] in G.nodes():
            G.nodes[row['target']]['party'] = row['target_party']
    
    if G.number_of_nodes() > 0 and nx.number_connected_components(G) > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G


# ============================================================================
# MAIN COLLECTION FUNCTIONS
# ============================================================================

def collect_period_metrics(df_period, period, topic, 
                           apply_backboning_func, calculate_modularity_func,
                           agreement_threshold, min_votes_threshold, backbone_alpha):
    """
    Collect all metrics for a single period using the provided functions.
    """
    metrics = {'period': period, 'topic': topic}
    
    if len(df_period) == 0:
        return metrics
    
    # --- RAW DATA STATS ---
    metrics['raw_edges'] = len(df_period)
    metrics['raw_nodes'] = len(set(df_period['source']) | set(df_period['target']))
    
    weights = df_period['weight'].values
    metrics['raw_weight_mean'] = np.mean(weights)
    metrics['raw_weight_median'] = np.median(weights)
    metrics['raw_weight_std'] = np.std(weights)
    metrics['raw_weight_min'] = np.min(weights)
    metrics['raw_weight_max'] = np.max(weights)
    
    # --- THRESHOLD GRAPH ---
    G_thresh = build_graph_with_threshold(df_period, agreement_threshold)
    
    basic = compute_basic_graph_stats(G_thresh)
    for k, v in basic.items():
        metrics[f'thresh_{k}'] = v
    
    cent = extract_centrality_stats_silent(G_thresh, "Threshold Network")
    for k, v in cent.items():
        metrics[f'thresh_{k}'] = v
    
    mod = compute_modularity_metrics(G_thresh, calculate_modularity_func)
    for k, v in mod.items():
        metrics[f'thresh_{k}'] = v
    
    assort = extract_assortativity_stats(G_thresh, f"Period {period}")
    for k, v in assort.items():
        metrics[f'thresh_{k}'] = v
    
    # --- BACKBONE GRAPH ---
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        df_bb, bb_stats, G_bb = apply_backboning_func(
            df_period,
            min_votes_threshold=min_votes_threshold,
            alpha=backbone_alpha
        )
        
        metrics['bb_original_nodes'] = bb_stats.get('original_nodes', 0)
        metrics['bb_original_edges'] = bb_stats.get('original_edges', 0)
        metrics['bb_after_filter_nodes'] = bb_stats.get('after_min_votes_filter_nodes', 0)
        metrics['bb_after_filter_edges'] = bb_stats.get('after_min_votes_filter_edges', 0)
        metrics['bb_backbone_nodes'] = bb_stats.get('backbone_nodes', 0)
        metrics['bb_backbone_edges'] = bb_stats.get('backbone_edges', 0)
        
        if G_bb.number_of_nodes() > 0:
            edges_to_remove = [(u, v) for u, v, attr in G_bb.edges(data=True) 
                               if attr.get('weight', 0) <= agreement_threshold]
            G_bb_thresh = G_bb.copy()
            G_bb_thresh.remove_edges_from(edges_to_remove)
            G_bb_thresh.remove_nodes_from(list(nx.isolates(G_bb_thresh)))
            
            metrics['bb_thresh_num_nodes'] = G_bb_thresh.number_of_nodes()
            metrics['bb_thresh_num_edges'] = G_bb_thresh.number_of_edges()
            
            bb_basic = compute_basic_graph_stats(G_bb_thresh)
            for k, v in bb_basic.items():
                metrics[f'bb_{k}'] = v
            
            bb_cent = extract_centrality_stats_silent(G_bb_thresh, "Backbone Network")
            for k, v in bb_cent.items():
                metrics[f'bb_{k}'] = v
            
            bb_mod = compute_modularity_metrics(G_bb_thresh, calculate_modularity_func)
            for k, v in bb_mod.items():
                metrics[f'bb_{k}'] = v
            
            bb_assort = extract_assortativity_stats(G_bb_thresh, f"Period {period} Backbone")
            for k, v in bb_assort.items():
                metrics[f'bb_{k}'] = v
                
    except Exception as e:
        print(f"Backbone error for period {period}: {e}", file=sys.stderr)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    return metrics


def collect_all_metrics(from_github_func, apply_backboning_func, calculate_modularity_func, 
                        periods, topics,
                        agreement_threshold,
                        min_votes_threshold,
                        backbone_alpha,
                        output_dir='./voting-data'):
    """
    Collect metrics for all periods and topics.
    """
    all_metrics = []
    
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"TOPIC: {topic}")
        print(f"{'='*60}")
        
        try:
            df_full = pd.read_csv(from_github_func(
                f"/edges/politician/Period/politician_edges_{topic}_by_Period.csv"
            ))
            print(f"Loaded {len(df_full)} total edges")
        except Exception as e:
            print(f"ERROR loading {topic}: {e}")
            continue
        
        for period in tqdm(periods, desc="Periods"):
            df_period = df_full[df_full['Period'] == period].copy()
            
            if len(df_period) == 0:
                print(f"  Period {period}: No data")
                all_metrics.append({'period': period, 'topic': topic})
                continue
            
            metrics = collect_period_metrics(
                df_period, period, topic, 
                apply_backboning_func, calculate_modularity_func,
                agreement_threshold, min_votes_threshold, backbone_alpha
            )
            all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(f'{output_dir}/period_metrics_all.csv', index=False)
        print(f"\nSaved to {output_dir}/period_metrics_all.csv")
        
        for topic in topics:
            topic_df = metrics_df[metrics_df['topic'] == topic].copy()
            safe_topic = topic.replace('ø', 'o').replace('æ', 'ae').replace('å', 'aa')
            topic_df.to_csv(f'{output_dir}/period_metrics_{safe_topic}.csv', index=False)
        print("Saved topic-specific files")
    
    return metrics_df


def print_metrics_summary(metrics_df):
    """Print a summary of the collected metrics."""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(f"\nCollected {len(metrics_df)} rows with {len(metrics_df.columns)} columns")
    
    key_cols = [
        'period', 'topic', 
        'raw_edges', 'raw_nodes', 
        'thresh_num_nodes', 'thresh_num_edges',
        'thresh_party_modularity', 'thresh_louvain_modularity',
        'thresh_party_assortativity',
        'bb_backbone_nodes', 'bb_backbone_edges',
        'bb_party_assortativity'
    ]
    
    existing_cols = [c for c in key_cols if c in metrics_df.columns]
    print("\nKEY METRICS PREVIEW:")
    print(metrics_df[existing_cols].to_string(index=False))


# ============================================================================
# DEFAULT RUN
# ============================================================================

if __name__ == "__main__":
    PERIODS = [66, 67, 68, 69, 70, 71]
    TOPICS = ['general', 'klima_miljo', 'immigration']
    AGREEMENT_THRESHOLD = 0.7
    MIN_VOTES_THRESHOLD = 10
    BACKBONE_ALPHA = 0.1

    metrics_df = collect_all_metrics(
        from_github_func=from_github,
        apply_backboning_func=apply_backboning,
        calculate_modularity_func=calculate_modularity,
        periods=PERIODS,
        topics=TOPICS,
        agreement_threshold=AGREEMENT_THRESHOLD,
        min_votes_threshold=MIN_VOTES_THRESHOLD,
        backbone_alpha=BACKBONE_ALPHA
    )

    metrics_df.to_csv('metrics_p66-71_general_klima_miljo_immigration.csv', index=False)
    print_metrics_summary(metrics_df)
