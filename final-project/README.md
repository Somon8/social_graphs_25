# Final Project - Danish Parliament Voting Analysis

This directory contains notebooks and data for analyzing voting patterns in the Danish Parliament (Folketinget) using data from the [Danish Parliament Open Data API](https://oda.ft.dk).

## Overview

The project analyzes voting records from Danish parliamentarians to understand:
- Voting agreement patterns between politicians
- Party cohesion and inter-party relationships
- Political network structures across different parliamentary periods

## Data Pipeline

The project follows a multi-stage data pipeline:

### 1. Data Collection
- **`Building votings dataframes.ipynb`**: Fetches all voting sessions (Afstemning) from the Danish Parliament API with meeting information
  - Retrieves 10,304 voting sessions
  - Enriches with case (Sag) information
  - Maps votes to parliamentary periods (65-71)
  - Outputs: `folketinget_votings_enriched.csv`, `votings.csv`

- **`voting_id_to_df_votes.ipynb`**: Converts voting session IDs into individual vote records
  - Fetches individual votes (Stemme) for each voting session
  - Extracts politician names, parties, and vote types from actor biographies
  - Processes votes for periods 65-71
  - Outputs: `voting-data/df_votes_p{period}.csv` and `voting-data/df_votes_all_periods.csv`

### 2. Data Enrichment
- **`df_votes_to_df_votes_enriched.ipynb`**: Enriches vote data with additional metadata
  - Merges individual votes with voting session metadata
  - Adds period information, meeting dates, and vote outcomes
  - Creates final enriched dataset with 1,007,700 vote records
  - Output: `voting-data/df_votes_enriched.csv`

### 3. Network Graph Creation
- **`df_enriched_to_graphs.ipynb`**: Constructs politician agreement networks
  - Calculates pairwise agreement percentages between politicians
  - Filters by period (currently focused on period 70: June 2019 - November 2022)
  - Creates network graphs with agreement threshold filtering
  - Uses ForceAtlas2 layout for visualization
  - Analyzes 168,800 votes from period 70

## Exploratory Notebooks

- **`thor_test.ipynb`**: Experimental analysis notebook featuring:
  - Party voting analysis classes
  - Party cohesion metrics (Rice Index, Agreement Index)
  - Inter-party agreement matrices
  - Coalition cohesion analysis
  - Voting bloc identification
  - Polarization index calculations

- **`Untitled-1.ipynb`**: Contains Danish Parliament API client implementation with comprehensive error handling and pagination support

## Data Files

### Core Data
- **`folketinget_votings_enriched.csv`**: 10,304 voting sessions with metadata (59 columns)
  - Voting conclusions, meeting information, case details, periods

- **`votings.csv`**: Simplified voting session data (22 columns)

- **`cases.csv`**: Parliamentary case information

### Voting Records
Located in `voting-data/`:
- **`df_votes_all_periods.csv`**: All individual votes across periods 65-71
- **`df_votes_enriched.csv`**: Enriched individual votes (1,007,700 rows, 7 columns)
  - Columns: voting_id, party, politician, vote_type, Period, vedtaget, Møde.dato
- **`df_votes_p{65-71}.csv`**: Individual vote records per parliamentary period

### Graph Data
Located in `graphs/`:
- Pre-computed graph data for network analysis

## Parliamentary Periods Covered

The data spans 7 recent parliamentary periods:
- **Period 65**: November 2001 - February 2005 (294 votes)
- **Period 66**: February 2005 - November 2007 (1,269 votes)
- **Period 67**: November 2007 - September 2011 (1,778 votes)
- **Period 68**: September 2011 - June 2015 (1,727 votes)
- **Period 69**: June 2015 - June 2019 (2,019 votes)
- **Period 70**: June 2019 - November 2022 (1,838 votes)
- **Period 71**: November 2022 - present (1,379 votes)

## Vote Types

Individual votes are coded as:
- **1**: For (støtter/ja)
- **2**: Against (imod/nej)
- **3**: Absent (fraværende)
- **4**: Abstain (undlader at stemme)

## Key Features

### Agreement Calculation
For each pair of politicians:
- Identifies shared voting sessions
- Counts votes where they agreed
- Calculates agreement percentage: `agree / total`
- Creates weighted edges in network graph

### Party Analysis
- Rice Index: Measures party voting cohesion (0-100%)
- Agreement Index: Alternative cohesion measure
- Inter-party agreement matrices
- Coalition stress point identification

### Network Visualization
- Node = Politician
- Edge weight = Agreement percentage
- Filtering by agreement threshold (e.g., >70%)
- Party color coding
- ForceAtlas2 layout algorithm

## Requirements

```python
networkx
pandas
matplotlib
requests
urllib
tqdm
```

## Usage Example

```python
# Load enriched voting data
import pandas as pd
df = pd.read_csv('voting-data/df_votes_enriched.csv')

# Filter by period
df_period_70 = df[df['Period'] == 70]

# Analyze agreement patterns
# (See df_enriched_to_graphs.ipynb for full implementation)
```

## Data Source

All data is fetched from the Danish Parliament Open Data API:
- Base URL: `https://oda.ft.dk/api/`
- Entities used: Afstemning (Voting), Stemme (Vote), Aktør (Actor), Sag (Case), Møde (Meeting)
- Documentation: [oda.ft.dk](https://oda.ft.dk)

## Notes

- The API client includes rate limiting (100ms between requests) and retry logic with exponential backoff
- Individual party affiliations are extracted from biographical XML data
- Some votes may have "Unknown" party classification if biography parsing fails
- Network graphs can be filtered by time period, topic, or agreement threshold for different analyses
