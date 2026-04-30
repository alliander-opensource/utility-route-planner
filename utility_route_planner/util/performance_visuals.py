# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
import numpy as np

# Data (copied from run logs)
cases = ["1", "2", "3", "4", "5"]
x = np.arange(len(cases))
width = 0.25  # Width of the bars

# Nodes per cell size
nodes_05 = [57657480, 71531445, 148564095, 22978462, 7911193]
nodes_5 = [576532, 715013, 1485900, 230019, 79114]
nodes_10 = [143926, 179037, 371583, 57491, 19783]

# CPU time in seconds
cpu_05 = [36.81, 62.15, 140.88, 16.32, 9.26]
cpu_5 = [3.32, 14.22, 18.5, 7.54, 4.82]
cpu_10 = [2.96, 12.23, 16.65, 6.96, 4.28]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Graph complexity (nodes)
rects1 = ax[0].bar(x - width, nodes_05, width, label="0.5m", color="#1f77b4", edgecolor="black", alpha=0.8)
rects2 = ax[0].bar(x, nodes_5, width, label="5m", color="#ff7f0e", edgecolor="black", alpha=0.8)
rects3 = ax[0].bar(x + width, nodes_10, width, label="10m", color="#2ca02c", edgecolor="black", alpha=0.8)

ax[0].set_ylabel("Graph nodes (log scale)", fontsize=10, fontweight="bold")
ax[0].set_xlabel("Case study", fontsize=10, fontweight="bold")
ax[0].set_title("Impact of cell size on graph size", fontsize=11)
ax[0].set_xticks(x)
ax[0].set_xticklabels(cases)
ax[0].set_yscale("log")  # Log for readability
ax[0].grid(axis="y", linestyle="--", alpha=0.5)
ax[0].legend()

# Plot 2: Processing time (CPU)
rects4 = ax[1].bar(x - width, cpu_05, width, label="0.5m", color="#1f77b4", edgecolor="black", alpha=0.8)
rects5 = ax[1].bar(x, cpu_5, width, label="5m", color="#ff7f0e", edgecolor="black", alpha=0.8)
rects6 = ax[1].bar(x + width, cpu_10, width, label="10m", color="#2ca02c", edgecolor="black", alpha=0.8)

ax[1].set_ylabel("CPU time (seconds, log scale)", fontsize=10, fontweight="bold")
ax[1].set_xlabel("Case study", fontsize=10, fontweight="bold")
ax[1].set_title("Impact of cell size on processing time", fontsize=11)
ax[1].set_xticks(x)
ax[1].set_xticklabels(cases)
ax[1].set_yscale("log")  # Log for readability
ax[1].grid(axis="y", linestyle="--", alpha=0.5)
ax[1].legend()

# Annotate subplots
ax[0].text(0.01, 0.99, "(a)", transform=ax[0].transAxes, fontsize=12, fontweight="bold", va="top")
ax[1].text(0.01, 0.99, "(b)", transform=ax[1].transAxes, fontsize=12, fontweight="bold", va="top")

plt.tight_layout()
plt.show()
