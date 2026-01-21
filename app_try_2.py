import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random

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
random.seed(42)

# ============================
# Neighborhood names
# ============================
NEIGHBORHOOD_NAMES = [
    "Mill Creek",
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
EMPTY, ROAD, HOUSE, SCHOOL, PARK, GROCERY, STORE, LIBRARY, OPENLAND = range(9)

# ============================
# Build city grid (visual only)
# ============================
city_grid = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# ----------------------------
# Roads
# ----------------------------
for _, r in roads.iterrows():
    city_grid[r["from_y"]][r["from_x"]] = ROAD
    city_grid[r["to_y"]][r["to_x"]] = ROAD

# ----------------------------
# Neighborhoods (3x2 blocks)
# ----------------------------
for _, n in neighborhoods.iterrows():
    for dx in [0, 1, 2]:
        for dy in [0, 1]:
            x, y = n["x"] + dx, n["y"] + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if city_grid[y][x] == EMPTY:
                    city_grid[y][x] = HOUSE

# ----------------------------
# Large parks (multi-tile)
# ----------------------------
parks = [
    (2, 2, 3, 4),
    (11, 3, 3, 3),
    (6, 9, 3, 4),
    (15, 6, 3, 3)
]
for px, py, w, h in parks:
    for dx in range(w):
        for dy in range(h):
            x, y = px + dx, py + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if city_grid[y][x] == EMPTY:
                    city_grid[y][x] = PARK

# ----------------------------
# Helper: check road adjacency
# ----------------------------
def is_road_adjacent(x, y):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if city_grid[ny][nx] == ROAD:
                return True
    return False

# ----------------------------
# School campus (3x3), moved left
# ----------------------------
SCHOOL_X, SCHOOL_Y = 14, 18
school_zone = [
    (13,17),(14,17),(15,17),
    (13,18),(14,18),(15,18),
    (13,19),(14,19),(15,19)
]
for x, y in school_zone:
    city_grid[y][x] = SCHOOL

# Park next to school
for x, y in [(12,17),(12,18),(12,19)]:
    if city_grid[y][x] == EMPTY:
        city_grid[y][x] = PARK

# ----------------------------
# Single library (2x2)
# ----------------------------
library_zone = [(9,12),(10,12),(9,13),(10,13)]
for x, y in library_zone:
    if city_grid[y][x] != ROAD:
        city_grid[y][x] = LIBRARY

# ----------------------------
# Realistic fill
# ----------------------------
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        if city_grid[y][x] != EMPTY:
            continue

        r = random.random()

        if is_road_adjacent(x, y):
            if r < 0.15:
                city_grid[y][x] = GROCERY
            elif r < 0.30:
                city_grid[y][x] = STORE
            elif r < 0.45:
                city_grid[y][x] = HOUSE
            else:
                city_grid[y][x] = OPENLAND
        else:
            if r < 0.20:
                city_grid[y][x] = PARK
            elif r < 0.30:
                city_grid[y][x] = HOUSE
            else:
                city_grid[y][x] = OPENLAND

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
        LIBRARY: "#8da0cb",
        SCHOOL: "#fff2cc",
        OPENLAND: "#d9f2d9"
    }

    label_map = {
        HOUSE: "H",
        PARK: "P",
        GROCERY: "G",
        STORE: "T",
        LIBRARY: "L"
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ax.add_patch(
                plt.Rectangle(
                    (x, GRID_SIZE - y - 1),
                    1, 1,
                    color=color_map[grid[y][x]],
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

    # School star
    ax.scatter(
        14.5,
        GRID_SIZE - 18.5,
        marker="*",
        s=500,
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
        Patch(facecolor="#8da0cb", label="Library (L)"),
        Patch(facecolor="#fff2cc", label="School Campus"),
        Patch(facecolor="#d9f2d9", label="Green Space"),
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
