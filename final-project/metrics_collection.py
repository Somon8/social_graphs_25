"""
Metrics Collection Across Parliamentary Periods

This module collects network metrics across all periods for specified topics
using the existing analysis functions.

Usage:
    from metrics_collection import collect_all_metrics, print_metrics_summary
    
    metrics_df = collect_all_metrics(
        from_github_func=from_github,
        apply_backboning_func=apply_backboning,
        calculate_modularity_func=calculate_modularity,
        periods=[66, 67, 68, 69, 70, 71],
        topics=['klima_miljø', 'immigration'],
        agreement_threshold=0.7,
        min_votes_threshold=10,
        backbone_alpha=0.1
    )
    
    print_metrics_summary(metrics_df)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import sys
import os

# Import your modules
import political_centrality_analysis
import assortativity_analysis


# ============================================================================
# WRAPPER FUNCTIONS
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


def extract_centrality_stats(centrality_dict):
    """
    Extract summary statistics from centrality dict returned by 
    political_centrality_analysis.calculate_centrality_measures().
    """
    stats = {}
    
    for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
        if centrality_dict.get(measure) is not None:
            values = list(centrality_dict[measure].values())
            measure_stats = summarize_distribution(values, measure)
            stats.update(measure_stats)
        else:
            for suffix in ['mean', 'median', 'max', 'min', 'std', 'variance', 'skewness', 'kurtosis']:
                stats[f'{measure}_{suffix}'] = np.nan
    
    return stats


def extract_assortativity_stats(G, period_label=""):
    """
    Wrapper around assortativity_analysis.analyze_assortativity() 
    that suppresses printing.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        results = assortativity_analysis.analyze_assortativity(G, period_label)
    except Exception as e:
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
    Wrapper around political_centrality_analysis.calculate_centrality_measures()
    that suppresses printing and returns summary stats.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        centrality_dict = political_centrality_analysis.calculate_centrality_measures(G, network_name)
        stats = extract_centrality_stats(centrality_dict)
    except Exception as e:
        stats = {}
        for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
            for suffix in ['mean', 'median', 'max', 'min', 'std', 'variance', 'skewness', 'kurtosis']:
                stats[f'{measure}_{suffix}'] = np.nan
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
        
        # Analyze community composition
        community_analysis = analyze_louvain_communities(G, louvain_communities)
        stats.update(community_analysis)
        
    except:
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
    
    Returns metrics about:
    - Size of largest community
    - Party diversity within communities
    - Whether communities map to traditional blocs
    """
    stats = {}
    total_nodes = G.number_of_nodes()
    
    # Sort communities by size
    sorted_communities = sorted(louvain_communities, key=len, reverse=True)
    
    # Analyze each community's party composition
    community_summaries = []
    
    for i, community in enumerate(sorted_communities):
        party_counts = defaultdict(int)
        for node in community:
            party = G.nodes[node].get('party', 'Unknown')
            party_counts[party] += 1
        
        # Sort parties by count
        sorted_parties = sorted(party_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create summary string: "V:15,S:12,M:8" etc
        summary = ','.join([f"{p}:{c}" for p, c in sorted_parties[:5]])  # Top 5 parties
        community_summaries.append(f"C{i}({len(community)}):[{summary}]")
        
        # Detailed stats for largest community
        if i == 0:
            stats['louvain_largest_community_size'] = len(community)
            stats['louvain_largest_community_pct'] = len(community) / total_nodes * 100
            stats['louvain_num_parties_in_largest'] = len(party_counts)
            
            dominant_party, dominant_count = sorted_parties[0]
            stats['louvain_dominant_party_in_largest'] = dominant_party
            stats['louvain_dominant_party_pct_in_largest'] = dominant_count / len(community) * 100
    
    # Full composition string (for inspection)
    stats['louvain_community_composition'] = ' | '.join(community_summaries)
    
    # Calculate party fragmentation across communities
    # (do parties stay together or split across communities?)
    stats['louvain_party_fragmentation'] = calculate_party_fragmentation(G, sorted_communities)
    
    return stats


def calculate_party_fragmentation(G, sorted_communities):
    """
    Calculate how fragmented parties are across Louvain communities.
    
    Returns a score 0-1 where:
    - 0 = all members of each party are in the same community (no fragmentation)
    - 1 = party members are evenly distributed across all communities (max fragmentation)
    """
    # Get all parties and their members
    party_members = defaultdict(list)
    for node, data in G.nodes(data=True):
        party = data.get('party', 'Unknown')
        party_members[party].append(node)
    
    # Create node -> community mapping
    node_to_community = {}
    for i, community in enumerate(sorted_communities):
        for node in community:
            node_to_community[node] = i
    
    # For each party, calculate what fraction of members are in each community
    fragmentation_scores = []
    
    for party, members in party_members.items():
        if len(members) < 2:
            continue
            
        # Count members per community
        community_counts = defaultdict(int)
        for member in members:
            comm = node_to_community.get(member, -1)
            community_counts[comm] += 1
        
        # Calculate concentration (1 - Herfindahl index normalized)
        # If all in one community: HHI = 1, fragmentation = 0
        # If evenly split: HHI approaches 1/n, fragmentation approaches 1
        total = len(members)
        hhi = sum((count / total) ** 2 for count in community_counts.values())
        
        # Normalize: fragmentation = 1 - HHI
        # (ranges from 0 to 1-1/n_communities)
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
    Matches the logic in edgelist_and_graph_to_visualization().
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
            df_full = pd.read_csv(from_github_func(f"/edges/politician/Period/politician_edges_{topic}_by_Period.csv"))
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
# PLOTTING FUNCTIONS
# ============================================================================

TOPIC_COLORS = {
    'general': '#666666',
    'klima_miljo': '#2ca02c',
    'klima_miljø': '#2ca02c', 
    'immigration': '#1f77b4'
}


def plot_metric_by_topic(metrics_df, col, title=None, ylabel=None, figsize=(12, 4), show_backbone=True):
    """
    One metric, three subplots (one per topic), threshold vs backbone.
    
    Parameters:
    -----------
    show_backbone : bool
        If True, show both threshold and backbone. If False, show only threshold.
    """
    topics = metrics_df['topic'].unique()
    fig, axes = plt.subplots(1, len(topics), figsize=figsize)
    if len(topics) == 1:
        axes = [axes]
    
    # Determine backbone column
    bb_col = None
    if show_backbone and col.startswith('thresh_'):
        bb_col = 'bb_' + col[7:]
    
    for ax, topic in zip(axes, topics):
        data = metrics_df[metrics_df['topic'] == topic].sort_values('period')
        periods = data['period'].values
        
        ax.plot(periods, data[col], 'o-', lw=2, ms=8, color='steelblue', label='Threshold')
        
        if bb_col and bb_col in data.columns:
            ax.plot(periods, data[bb_col], 's--', lw=2, ms=8, color='coral', label='Backbone')
        
        ax.set_title(topic, fontweight='bold')
        ax.set_xlabel('Period')
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel(ylabel or col)
    fig.suptitle(title or col, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, axes


def plot_metric_all_topics_overlay(metrics_df, thresh_col, bb_col=None, title=None, ylabel=None, 
                                    figsize=(10, 6), show_backbone=True):
    """
    All topics on same plot. Solid=threshold, dashed=backbone.
    
    Parameters:
    -----------
    show_backbone : bool
        If True, show both threshold and backbone. If False, show only threshold.
    """
    fig, ax = plt.subplots(figsize=figsize)
    topics = metrics_df['topic'].unique()
    periods = sorted(metrics_df['period'].unique())
    
    for topic in topics:
        data = metrics_df[metrics_df['topic'] == topic].sort_values('period')
        color = TOPIC_COLORS.get(topic, 'black')
        
        ax.plot(data['period'], data[thresh_col], 'o-', lw=2, ms=8, 
                color=color, label=f'{topic} (thresh)')
        
        if show_backbone and bb_col and bb_col in data.columns:
            ax.plot(data['period'], data[bb_col], 's--', lw=2, ms=8,
                    color=color, alpha=0.6, label=f'{topic} (backbone)')
    
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel(ylabel or thresh_col, fontsize=11)
    ax.set_title(title or thresh_col, fontsize=13, fontweight='bold')
    ax.set_xticks(periods)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax


def plot_summary_grid(metrics_df, figsize=(16, 12), show_backbone=True):
    """
    2x3 grid of key metrics. All topics overlaid, threshold vs backbone.
    
    Parameters:
    -----------
    show_backbone : bool
        If True, show both threshold and backbone. If False, show only threshold.
    """
    metrics_to_plot = [
        ('thresh_num_edges', 'bb_backbone_edges', 'Edge Count'),
        ('thresh_party_modularity', 'bb_party_modularity', 'Party Modularity'),
        ('thresh_party_assortativity', 'bb_party_assortativity', 'Party Assortativity'),
        ('thresh_louvain_modularity', 'bb_louvain_modularity', 'Louvain Modularity'),
        ('thresh_degree_mean', 'bb_degree_mean', 'Mean Degree Centrality'),
        ('thresh_betweenness_mean', 'bb_betweenness_mean', 'Mean Betweenness Centrality'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    topics = metrics_df['topic'].unique()
    periods = sorted(metrics_df['period'].unique())
    
    for ax, (thresh_col, bb_col, title) in zip(axes, metrics_to_plot):
        for topic in topics:
            data = metrics_df[metrics_df['topic'] == topic].sort_values('period')
            color = TOPIC_COLORS.get(topic, 'black')
            
            if thresh_col in data.columns:
                ax.plot(data['period'], data[thresh_col], 'o-', lw=2, ms=6,
                        color=color, label=f'{topic} (thresh)')
            
            if show_backbone and bb_col in data.columns:
                ax.plot(data['period'], data[bb_col], 's--', lw=2, ms=6,
                        color=color, alpha=0.6, label=f'{topic} (bb)')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Period')
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3)
    
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=6, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def plot_all_metrics_vertical(metrics_df, figsize=(14, 36), show_backbone=True):
    """
    Vertical stack of individual metric plots. Each row is one metric with all topics overlaid.
    
    Parameters:
    -----------
    show_backbone : bool
        If True, show both threshold and backbone. If False, show only threshold.
    """
    metrics_to_plot = [
        ('raw_edges', None, 'Raw Edge Count'),
        ('thresh_num_edges', 'bb_backbone_edges', 'Edge Count (filtered)'),
        ('thresh_num_nodes', 'bb_backbone_nodes', 'Node Count'),
        ('thresh_density', 'bb_density', 'Network Density'),
        ('thresh_avg_clustering', 'bb_avg_clustering', 'Average Clustering'),
        ('thresh_party_modularity', 'bb_party_modularity', 'Party Modularity'),
        ('thresh_louvain_modularity', 'bb_louvain_modularity', 'Louvain Modularity'),
        ('thresh_party_assortativity', 'bb_party_assortativity', 'Party Assortativity'),
        ('thresh_degree_assortativity', 'bb_degree_assortativity', 'Degree Assortativity'),
        ('thresh_degree_mean', 'bb_degree_mean', 'Degree Centrality (mean)'),
        ('thresh_betweenness_mean', 'bb_betweenness_mean', 'Betweenness Centrality (mean)'),
        ('thresh_closeness_mean', 'bb_closeness_mean', 'Closeness Centrality (mean)'),
        ('thresh_eigenvector_mean', 'bb_eigenvector_mean', 'Eigenvector Centrality (mean)'),
    ]
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    topics = metrics_df['topic'].unique()
    periods = sorted(metrics_df['period'].unique())
    
    for ax, (thresh_col, bb_col, title) in zip(axes, metrics_to_plot):
        for topic in topics:
            data = metrics_df[metrics_df['topic'] == topic].sort_values('period')
            color = TOPIC_COLORS.get(topic, 'black')
            
            if thresh_col in data.columns:
                ax.plot(data['period'], data[thresh_col], 'o-', lw=2, ms=7,
                        color=color, label=f'{topic} (thresh)')
            
            if show_backbone and bb_col and bb_col in data.columns:
                ax.plot(data['period'], data[bb_col], 's--', lw=2, ms=7,
                        color=color, alpha=0.6, label=f'{topic} (bb)')
        
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=3)
    
    axes[-1].set_xlabel('Period', fontsize=11)
    
    subtitle = '(solid = threshold, dashed = HSS backbone)' if show_backbone else '(threshold only)'
    fig.suptitle(f'All Metrics Over Parliamentary Periods\n{subtitle}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes


def plot_single_row(metrics_df, metrics_list, figsize=(16, 4), show_backbone=True):
    """
    Single row of subplots, one per metric. All topics overlaid on each.
    
    Parameters:
    -----------
    metrics_list : list of tuples
        Each tuple: (thresh_col, bb_col, title)
    show_backbone : bool
        If True, show both threshold and backbone. If False, show only threshold.
    """
    n = len(metrics_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    topics = metrics_df['topic'].unique()
    periods = sorted(metrics_df['period'].unique())
    
    for ax, (thresh_col, bb_col, title) in zip(axes, metrics_list):
        for topic in topics:
            data = metrics_df[metrics_df['topic'] == topic].sort_values('period')
            color = TOPIC_COLORS.get(topic, 'black')
            
            if thresh_col in data.columns:
                ax.plot(data['period'], data[thresh_col], 'o-', lw=2, ms=7,
                        color=color, label=f'{topic}')
            
            if show_backbone and bb_col and bb_col in data.columns:
                ax.plot(data['period'], data[bb_col], 's--', lw=2, ms=7,
                        color=color, alpha=0.5)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Period')
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3)
    
    axes[0].legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig, axes