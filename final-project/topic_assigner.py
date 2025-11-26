"""
Folketinget Topic Classification Pipeline
=========================================
Classifies parliamentary cases by policy topic using stemmed keyword matching.
Produces exploratory statistics and visualizations.
"""
from nltk.stem.snowball import SnowballStemmer
import urllib.request
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

stemmer = SnowballStemmer("danish")


# =============================================================================
# STEMMING FUNCTIONS
# =============================================================================

def stem_word(word):
    """Stem a single Danish word"""
    return stemmer.stem(word.lower())

def stem_keyword_dict(keyword_dict):
    """Stem all keywords in the topic dictionary"""
    stemmed = {}
    for topic, words in keyword_dict.items():
        stemmed[topic] = list(set([stem_word(w) for w in words]))
    return stemmed

def tokenize_and_stem(text):
    """
    Tokenize Danish text and stem each token.
    Returns a set of unique stemmed tokens.
    """
    if pd.isna(text) or text is None:
        return set()
    
    # Lowercase and extract Danish words (including æ, ø, å)
    tokens = re.findall(r'[a-zæøå]+', text.lower())
    
    # Stem each token
    stemmed = [stem_word(t) for t in tokens if len(t) > 2]  # Skip very short tokens
    
    return set(stemmed)

# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_case(text, stemmed_keywords, use_substring=True):
    """
    Classify a case by counting keyword matches per topic.
    
    Parameters:
    -----------
    text : str
        The case description (resume + title)
    stemmed_keywords : dict
        Topic -> list of stemmed keywords
    use_substring : bool
        If True, also check for substring matches (helps with compound words)
    
    Returns:
    --------
    dict : topic -> match count
    """
    stemmed_tokens = tokenize_and_stem(text)
    
    scores = {}
    matched_keywords = {}  # For debugging/inspection
    
    for topic, stems in stemmed_keywords.items():
        matches = []
        
        for stem in stems:
            # Direct match
            if stem in stemmed_tokens:
                matches.append(stem)
            # Substring match for compound words (if enabled)
            elif use_substring:
                for token in stemmed_tokens:
                    if len(stem) >= 4 and stem in token:
                        matches.append(f"{stem}(in:{token})")
                        break
        
        scores[topic] = len(matches)
        matched_keywords[topic] = matches
    
    return scores, matched_keywords

def assign_topics(scores, threshold=1):
    """
    Assign topic labels based on scores.
    
    Returns:
    --------
    primary_topic : str or None
        The highest-scoring topic (None if no matches)
    all_topics : list
        All topics meeting the threshold
    """
    # Filter to topics meeting threshold
    qualifying = {t: s for t, s in scores.items() if s >= threshold}
    
    if not qualifying:
        return None, []
    
    # Primary = highest score
    primary = max(qualifying, key=qualifying.get)
    all_topics = list(qualifying.keys())
    
    return primary, all_topics

# =============================================================================
# BATCH CLASSIFICATION
# =============================================================================

def classify_all_cases(df, text_column, stemmed_keywords, threshold=1):
    """
    Classify all cases in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing cases
    text_column : str
        Name of column containing text to classify (e.g., 'resume' or combined)
    stemmed_keywords : dict
        Stemmed keyword dictionary
    threshold : int
        Minimum keyword matches to assign a topic
    
    Returns:
    --------
    pd.DataFrame : Original df with added classification columns
    """
    results = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        scores, matched = classify_case(text, stemmed_keywords)
        primary, all_topics = assign_topics(scores, threshold)
        
        results.append({
            'primary_topic': primary,
            'all_topics': all_topics,
            'topic_count': len(all_topics),
            'total_matches': sum(scores.values()),
            **{f'score_{t}': s for t, s in scores.items()}
        })
    
    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

# =============================================================================
# EXPLORATORY STATISTICS
# =============================================================================

def compute_coverage_stats(df):
    """Compute coverage statistics"""
    total = len(df)
    classified = (df['primary_topic'].notna()).sum()
    unclassified = (df['primary_topic'].isna()).sum()
    multi_topic = (df['topic_count'] > 1).sum()
    
    stats = {
        'total_cases': total,
        'classified': classified,
        'classified_pct': 100 * classified / total,
        'unclassified': unclassified,
        'unclassified_pct': 100 * unclassified / total,
        'multi_topic': multi_topic,
        'multi_topic_pct': 100 * multi_topic / total
    }
    
    return stats

def compute_topic_stats(df, topics):
    """Compute per-topic statistics"""
    stats = []
    
    for topic in topics:
        # Cases where this is primary topic
        primary_count = (df['primary_topic'] == topic).sum()
        
        # Cases where this topic appears at all
        any_count = df['all_topics'].apply(lambda x: topic in x if x else False).sum()
        
        # Average score when topic is present
        score_col = f'score_{topic}'
        if score_col in df.columns:
            avg_score = df[df[score_col] > 0][score_col].mean()
        else:
            avg_score = 0
        
        stats.append({
            'topic': topic,
            'primary_count': primary_count,
            'any_count': any_count,
            'avg_score_when_present': round(avg_score, 2) if not pd.isna(avg_score) else 0
        })
    
    return pd.DataFrame(stats).sort_values('primary_count', ascending=False)

def compute_overlap_matrix(df, topics):
    """Compute topic co-occurrence matrix"""
    matrix = pd.DataFrame(0, index=topics, columns=topics)
    
    for _, row in df.iterrows():
        if row['all_topics']:
            for t1 in row['all_topics']:
                for t2 in row['all_topics']:
                    if t1 in topics and t2 in topics:
                        matrix.loc[t1, t2] += 1
    
    return matrix

def compute_period_distribution(df, period_column, topics):
    """Compute topic distribution across periods"""
    # Cross-tabulation of primary topic by period
    crosstab = pd.crosstab(df[period_column], df['primary_topic'])
    
    # Ensure all topics are present
    for topic in topics:
        if topic not in crosstab.columns:
            crosstab[topic] = 0
    
    return crosstab[topics]

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_topic_distribution(topic_stats, figsize=(12, 6)):
    """Bar chart of cases per topic"""
    fig, ax = plt.subplots(figsize=figsize)
    
    topics = topic_stats['topic']
    counts = topic_stats['primary_count']
    
    bars = ax.bar(range(len(topics)), counts, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Cases per Topic (Primary Classification)')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_overlap_heatmap(overlap_matrix, figsize=(10, 8)):
    """Heatmap of topic co-occurrence"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize by diagonal (self-co-occurrence = topic count)
    # This shows proportion of overlap
    diag = np.diag(overlap_matrix.values)
    normalized = overlap_matrix.values / diag[:, np.newaxis]
    normalized = np.nan_to_num(normalized)  # Handle division by zero
    
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=overlap_matrix.columns,
                yticklabels=overlap_matrix.index,
                ax=ax)
    ax.set_title('Topic Overlap Matrix\n(Row topic → proportion also in column topic)')
    plt.tight_layout()
    return fig

def plot_period_distribution(period_dist, figsize=(14, 6)):
    """Stacked area chart of topics over periods"""
    fig, ax = plt.subplots(figsize=figsize)
    
    period_dist.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_xlabel('Period')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Topic Distribution Across Parliamentary Periods')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_match_strength_histogram(df, figsize=(10, 5)):
    """Histogram of keyword match counts"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Total matches
    axes[0].hist(df['total_matches'], bins=20, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Total Keyword Matches')
    axes[0].set_ylabel('Number of Cases')
    axes[0].set_title('Distribution of Match Strength')
    axes[0].axvline(df['total_matches'].median(), color='red', linestyle='--', 
                    label=f"Median: {df['total_matches'].median():.0f}")
    axes[0].legend()
    
    # Topic count
    topic_counts = df['topic_count'].value_counts().sort_index()
    axes[1].bar(topic_counts.index, topic_counts.values, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Number of Topics Matched')
    axes[1].set_ylabel('Number of Cases')
    axes[1].set_title('Multi-Topic Classification')
    
    plt.tight_layout()
    return fig