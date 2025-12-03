def create_party_color_mapping(edgelist):
    """
    Create a mapping from politician names to party colors based on the edgelist.
    
    Parameters:
    -----------
    edgelist : pandas.DataFrame
        DataFrame with columns: source, target, source_party, target_party
    
    Returns:
    --------
    tuple : (politician_to_color dict, party_colors dict)
    """
    # Define party colors (Danish political parties)
    party_colors = {
    # Main Danish parties - sourced from Wikidata/Wikipedia
    'Socialdemokratiet': "#C82518", 
    'Venstre': '#002883',                  # Wikidata - dark blue
    'Det Konservative Folkeparti': '#00583C',  # Dark green (Conservative)
    'Socialistisk Folkeparti': '#E4007C',  # Magenta/Pink (SF - Green Left)
    'Radikale Venstre': '#E52483',         # Pink/Magenta (Social Liberals)
    'Enhedslisten': '#E6801A',             # Red-Orange (Red-Green Alliance) 
    'Dansk Folkeparti': '#FFCD00',         # Yellow (Danish People's Party)
    'Liberal Alliance': '#13B5EA',         # Light/Cyan blue
    'Alternativet': '#00FF00',             # Bright Green
    'Nye Borgerlige': '#004450',           # Dark teal (New Right)
    'Kristendemokraterne': '#8B8D8E',      # Grey (Christian Democrats)
    'Frie Grønne': '#3AAA35',              # Green
    'Danmarksdemokraterne': '#004B87',     # Blue (Denmark Democrats)
    'Moderaterne': '#8B2346',              # Purple-maroon
    
    # Greenlandic parties
    'Siumut': '#C8102E',                   # Red
    'Inuit Ataqatigiit': '#44712E',        # Green
    'Naleraq': '#005B82',                  # Blue
    'Nunatta Qitornai': '#FDB913',         # Yellow/Gold
    
    # Faroese parties  
    'Javnaðarflokkurin': '#E30613',        # Red (Social Democrats)
    'Sambandsflokkurin': '#003F87',        # Blue (Union Party)
    'Tjóðveldi': '#005DAA',                # Blue (Republic)
    'Folkaflokkurin': '#007A3D',           # Green (People's Party)
}
    
    # Build politician to party mapping from both source and target
    politician_to_party = {}
    
    for _, row in edgelist.iterrows():
        politician_to_party[row['source']] = row['source_party']
        politician_to_party[row['target']] = row['target_party']
    
    # Map politicians to colors
    politician_to_color = {
        politician: party_colors.get(party, '#808080')  # Default gray for unknown parties
        for politician, party in politician_to_party.items()
    }
    
    return politician_to_color, party_colors
