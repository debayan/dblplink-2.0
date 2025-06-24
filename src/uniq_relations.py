relations = set()

with open("dblp.nt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        try:
            parts = line.strip().rstrip(" .").split(" ", 2)
            if len(parts) < 3:
                continue
            _, p, _ = parts
            if p.startswith("<") and p.endswith(">"):
                relations.add(p.strip("<>"))
        except Exception:
            continue

# Print sorted list
for rel in sorted(relations):
    print(rel)

