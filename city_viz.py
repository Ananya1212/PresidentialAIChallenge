import streamlit as st

def draw_city(path, G, RISK_COLOR):
    st.error("🔥 draw_city() IS RUNNING 🔥")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import streamlit as st

def draw_city(path, G, RISK_COLOR):
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---------------------------
    # City layout (fixed grid)
    # ---------------------------
    city = {
        "North Park": (2, 6),
        "River Heights": (4, 5),
        "Maplewood": (3, 3),
        "Oak Valley": (1, 2),
        "Central High School": (5, 3)
    }

    # ---------------------------
    # Draw roads
    # ---------------------------
    for x in range(0, 7):
        ax.plot([x, x], [0, 7], color="#cccccc", linewidth=1)
    for y in range(0, 8):
        ax.plot([0, 6], [y, y], color="#cccccc", linewidth=1)

    # ---------------------------
    # Draw houses
    # ---------------------------
    for (x, y) in city.values():
        for dx in [-0.3, 0.3]:
            for dy in [-0.3, 0.3]:
                ax.add_patch(
                    Rectangle((x + dx - 0.15, y + dy - 0.15),
                              0.25, 0.25,
                              facecolor="#dddddd",
                              edgecolor="black",
                              linewidth=0.5)
                )

    # ---------------------------
    # Draw neighborhoods & school
    # ---------------------------
    for name, (x, y) in city.items():
        if "School" in name:
            ax.add_patch(Rectangle((x - 0.4, y - 0.4),
                                   0.8, 0.8,
                                   facecolor="#4CAF50"))
        ax.text(x, y + 0.6, name,
                ha='center', fontsize=10, weight='bold')

    # ---------------------------
    # Draw route arrows
    # ---------------------------
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        x1, y1 = city[start]
        x2, y2 = city[end]

        risk = G[start][end]["accident"]
        color = [c / 255 for c in RISK_COLOR[risk]]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle='->',
            linewidth=6,
            color=color,
            mutation_scale=20
        )
        ax.add_patch(arrow)

    # ---------------------------
    # Final formatting
    # ---------------------------
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Illustrative Example City Route", fontsize=16)
    ax.set_aspect("equal")
    ax.axis("off")

    st.pyplot(fig)
