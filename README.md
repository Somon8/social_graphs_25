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
import urllib
for node in DG.nodes():
    filename = urllib.parse.quote(node, safe = "")
    with open(f"rock_data/{filename}.txt", "r", encoding="utf-8") as f:
        content = f.read()
```

Get the node_df, a dataframe containing nodes as index, and some metrics/measurements as columns:

```
df = pd.read_csv(from_github("graphs/node_df.csv"), index_col = "Node")
```
