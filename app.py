import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================
# App title
# ============================
st.title("SafeFlow AI 🚦")
st.subheader("AI-powered school route optimization")

# ============================
# Load trained ML model
# ============================
with open("safeflow_speed_model.pkl", "rb") as f:
    automl = pickle.load(f)

# ============================
# Load city data
# ============================
roads = pd.read_csv("roads_raw.csv")
neighborhoods = pd.read_csv("neighborhoods.csv")

GRID_SIZE = 20
ROAD_DISTANCE_KM = 0.1

# ============================
# Human-readable neighborhood names
# ============================
NEIGHBORHOOD_NAMES = [
    "Panther Creek",
    "Maple Grove",
    "Riverstone",
    "Oak Ridge",
    "Sunset Hills",
    "Willow Bend",
    "Cedar Park",
    "Lakeside"
]
neighborhoods["display_name"] = NEIGHBORHOOD_NAMES[:len(neighborhoods)]

# ============================
# Cell types
# ============================
EMPTY, ROAD, HOUSE, SCHOOL, PARK, GROCERY, STORE = range(7)

# ============================
# Build city grid (VISUAL ONLY)
# ============================
city_grid = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Roads
for _, r in roads.iterrows():
    city_grid[r["from_y"]][r["from_x"]] = ROAD
    city_grid[r["to_y"]][r["to_x"]] = ROAD

# Neighborhood housing clusters (2x2)
for _, n in neighborhoods.iterrows():
    for dx in [0, 1]:
        for dy in [0, 1]:
            x, y = n["x"] + dx, n["y"] + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if city_grid[y][x] == EMPTY:
                    city_grid[y][x] = HOUSE

# Large parks (3x4)
parks = [(2, 2), (11, 3), (6, 9)]
for px, py in parks:
    for dx in range(3):
        for dy in range(4):
            x, y = px + dx, py + dy
            if city_grid[y][x] == EMPTY:
                city_grid[y][x] = PARK

# Commercial store clusters
for cx, cy in [(13, 11), (4, 14)]:
    for dx in [0, 1]:
        for dy in [0, 1]:
            x, y = cx + dx, cy + dy
            if city_grid[y][x] == EMPTY:
                city_grid[y][x] = STORE

# Grocery stores (more, realistic spacing)
grocery_locations = [
    (8, 6),
    (15, 9),
    (5, 3),
    (12, 5),
    (3, 10),
    (10, 14)
]
for gx, gy in grocery_locations:
    if city_grid[gy][gx] != ROAD:
        city_grid[gy][gx] = GROCERY

# School campus (2x2, directly next to road)
SCHOOL_X, SCHOOL_Y = 17, 18
school_zone = [(16,17), (17,17), (16,18), (17,18)]
for x, y in school_zone:
    if city_grid[y][x] != ROAD:
        city_grid[y][x] = SCHOOL

# Fill remaining empty cells with housing
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        if city_grid[y][x] == EMPTY:
            city_grid[y][x] = HOUSE

# ============================
# Visualization
# ============================
def draw_city(grid, path=None):
    color_map = {
        ROAD: "#bdbdbd",
        HOUSE: "#ffcc99",
        PARK: "#99cc99",
        GROCERY: "#66b2ff",
        STORE: "#cc99ff",
        SCHOOL: "#fff2cc"
    }

    label_map = {
        HOUSE: "H",
        PARK: "P",
        GROCERY: "G",
        STORE: "T"
    }

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ax.add_patch(
                plt.Rectangle(
                    (x, GRID_SIZE - y - 1),
                    1, 1,
                    color=color_map.get(grid[y][x], "#f5f5f5"),
                    ec="white"
                )
            )

            if grid[y][x] in label_map:
                ax.text(
                    x + 0.5,
                    GRID_SIZE - y - 0.5,
                    label_map[grid[y][x]],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold"
                )

    # School star (centered on campus)
    ax.scatter(
        16.5,
        GRID_SIZE - 17.5,
        marker="*",
        s=450,
        color="gold",
        edgecolors="black",
        zorder=6
    )

    # Route
    if path:
        coords = [(int(p[1:-1].split(",")[0]), int(p[1:-1].split(",")[1])) for p in path]
        xs = [x + 0.5 for x, y in coords]
        ys = [GRID_SIZE - y - 0.5 for x, y in coords]
        ax.plot(xs, ys, color="red", linewidth=3)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("SafeFlow AI – City Map & Optimal Route")

    legend_elements = [
        Patch(facecolor="#bdbdbd", label="Road"),
        Patch(facecolor="#ffcc99", label="Neighborhood (H)"),
        Patch(facecolor="#99cc99", label="Park (P)"),
        Patch(facecolor="#66b2ff", label="Grocery (G)"),
        Patch(facecolor="#cc99ff", label="Store (T)"),
        Patch(facecolor="#fff2cc", label="School Campus"),
        Patch(facecolor="red", label="Optimal Route")
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    return fig

# ============================
# Helper: snap to nearest road
# ============================
def find_nearest_road_node(nx0, ny0, roads_df):
    best, best_dist = None, 1e9
    for _, r in roads_df.iterrows():
        for x, y in [(r["from_x"], r["from_y"]), (r["to_x"], r["to_y"])]:
            d = abs(nx0 - x) + abs(ny0 - y)
            if d < best_dist:
                best, best_dist = (x, y), d
    return f"({best[0]},{best[1]})"

# ============================
# User inputs
# ============================
hour = st.slider("Hour of day", 0, 23, 8)
weather = st.selectbox("Weather", ["clear", "rain", "fog"])
priority = st.radio("Route preference", ["Shortest distance", "Least congestion", "Balanced"])

start_name = st.selectbox("Choose neighborhood", neighborhoods["display_name"])
start_row = neighborhoods[neighborhoods["display_name"] == start_name].iloc[0]

start_node = find_nearest_road_node(start_row["x"], start_row["y"], roads)
school_node = find_nearest_road_node(SCHOOL_X, SCHOOL_Y, roads)

# ============================
# Build routing graph
# ============================
G = nx.Graph()

for _, road in roads.iterrows():
    features = {
        "hour": hour,
        "day_of_week": 1,
        "is_school_day": 1,
        "is_arrival_time": int(7 <= hour <= 9),
        "is_dismissal_time": int(14 <= hour <= 16),
        "weather_condition": weather,
        "precipitation": int(weather == "rain"),
        "visibility_level": "low" if weather in ["rain", "fog"] else "high",
        "num_lanes": 2,
        "speed_limit": 30,
        "distance_km": ROAD_DISTANCE_KM,
        "is_intersection": 0,
        "neighborhood_population": 5000,
        "working_population_pct": 0.6,
        "students_population": 800,
        "distance_to_school_m": abs(road["to_x"] - SCHOOL_X) * 100
    }

    X = pd.get_dummies(pd.DataFrame([features]))
    X = X.reindex(columns=automl.feature_names_in_, fill_value=0)

    speed = max(5.0, automl.predict(X)[0])
    travel_time = ROAD_DISTANCE_KM / speed

    if priority == "Shortest distance":
        weight = ROAD_DISTANCE_KM
    elif priority == "Least congestion":
        weight = travel_time
    else:
        weight = 0.5 * ROAD_DISTANCE_KM + 0.5 * travel_time

    G.add_edge(
        f"({road['from_x']},{road['from_y']})",
        f"({road['to_x']},{road['to_y']})",
        weight=weight
    )

# ============================
# Run routing + visualize
# ============================
if st.button("Find best route"):
    path = nx.shortest_path(G, start_node, school_node, weight="weight")
    cost = nx.shortest_path_length(G, start_node, school_node, weight="weight")

    st.success("Best route found!")
    st.write(f"From **{start_name}** to **School**")
    st.write(f"Total route cost: {cost:.4f}")

    fig = draw_city(city_grid, path)
    st.pyplot(fig)
