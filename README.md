get the data using:
```python
def from_github(path):
    base_url = "https://raw.githubusercontent.com/Somon8/social_graphs_25/main/"
    url_to_file = base_url + path
    return urllib.request.urlopen(url_to_file)

#Example
DG = nx.read_gexf(from_github("graphs/rock_bands_graph.gexf"))
```

The downloaded wikipedia pages are saved in a UTF-encoded filename. The wikipedia pages of the files can be read using the following:

```python
github_base_url = "https://raw.githubusercontent.com/Somon8/social_graphs_25/main/"
def url_for_node(node: str):
    name_encoded = urllib.parse.quote(node, safe="")
    path_encoded = urllib.parse.quote(name_encoded, safe="")  #It's a bit messy but whatever
    return f"{github_base_url}rock_data/{path_encoded}.txt"

for node in DG.nodes():
    url = url_for_node(node)
    with urllib.request.urlopen(url) as r:
        content = r.read().decode("utf-8")
```

Get the node_df, a dataframe containing nodes as index, and some metrics/measurements as columns:

```
df = pd.read_csv(from_github("graphs/node_df.csv"), index_col = "Node")
```
