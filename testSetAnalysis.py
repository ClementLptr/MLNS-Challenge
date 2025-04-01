from collections import Counter, defaultdict

import matplotlib.pyplot as plt

with open("data/test.txt", "r") as f:
    lines = f.readlines()
    lines = [line.split() for line in lines]
    edges = [(int(edge[0]), int(edge[1])) for edge in lines]

degrees = defaultdict(int)
for edge in edges:
    degrees[edge[0]] += 1
    degrees[edge[1]] += 1


# Count frequency of each degree
degree_counts = Counter(degrees.values())

# Sort by degree for plotting
degrees_x = sorted(degree_counts.keys())
counts_y = [degree_counts[d] for d in degrees_x]

plt.figure(figsize=(10, 6))
plt.loglog(degrees_x, counts_y, "o-")
plt.xlabel("Degree (log scale)")
plt.ylabel("Number of nodes (log scale)")
plt.title("Degree Distribution (Log-Log Scale)")
plt.grid(True, alpha=0.3, which="both")
plt.savefig("test_set_node_occurences.png", dpi=300)

plt.show()
