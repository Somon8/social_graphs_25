The downloaded wikipedia pages are saved in a UTF-encoded filename. The wikipedia pages of the files can be read using the following:

```python
import urllib
for node in DG.nodes():
    filename = urllib.parse.quote(node, safe = "")
    with open(f"rock_data/{filename}.txt", "r", encoding="utf-8") as f:
        content = f.read()
```
