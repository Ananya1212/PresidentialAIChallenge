import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
# City layout (visual only)
# ============================
EMPTY, ROAD, HOUSE, SCHOOL, PARK, GROCERY, STORE = range(7)

city_grid = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

for _, r in roads.iterrows():
    city_grid[r["from_y"]][r["from_x"]] = ROAD
    city_grid[r["to_y"]][r["to_x"]] = ROAD

for _, n in neighborhoods.iterrows():
    city_grid[n["y"]][n["x"]] = HOUSE

city_grid[18][18] = SCHOOL

for x, y in [(4,4), (10,10), (15,3)]:
    city_grid[y][x] = PARK

for x, y in [(6,14), (12,6)]:
    city_grid[y][x] = GROCERY

for x, y in [(3,12), (14,14)]:
    city_grid[y][x] = STORE

# ============================
# Visualization helper
# ============================
def draw_city(grid, path=None):
    color_map = {
        EMPTY: "#f0f0f0",
        ROAD: "#bdbdbd",
        HOUSE: "#ffcc99",
        SCHOOL: "#ff6666",
        PARK: "#99cc99",
        GROCERY: "#66b2ff",
        STORE: "#cc99ff"
    }

    fig, ax = plt.subplots(figsize=(6, 6))

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

    if path:
        coords = [
            (int(p[1:-1].split(",")[0]), int(p[1:-1].split(",")[1]))
            for p in path
        ]
        xs = [x + 0.5 for x, y in coords]
        ys = [GRID_SIZE - y - 0.5 for x, y in coords]
        ax.plot(xs, ys, color="red", linewidth=3)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("SafeFlow AI – City Map & Optimal Route")

    return fig

# ============================
# Helper: snap to nearest road
# ============================
def find_nearest_road_node(nx0, ny0, roads_df):
    min_dist = float("inf")
    nearest = None

    for _, r in roads_df.iterrows():
        for x, y in [(r["from_x"], r["from_y"]), (r["to_x"], r["to_y"])]:
            d = abs(nx0 - x) + abs(ny0 - y)
            if d < min_dist:
                min_dist = d
                nearest = (x, y)

    return f"({nearest[0]},{nearest[1]})"

# ============================
# User inputs
# ============================
hour = st.slider("Hour of day", 0, 23, 8)
weather = st.selectbox("Weather", ["clear", "rain", "fog"])
priority = st.radio("Route preference", ["Shortest distance", "Least congestion", "Balanced"])

start_neighborhood_id = st.selectbox(
    "Choose neighborhood",
    neighborhoods["neighborhood_id"].tolist()
)

start_row = neighborhoods[neighborhoods["neighborhood_id"] == start_neighborhood_id].iloc[0]
start_node = find_nearest_road_node(start_row["x"], start_row["y"], roads)
school_node = find_nearest_road_node(18, 18, roads)

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
        "distance_to_school_m": abs(road["to_x"] - 18) * 100
    }

    X = pd.DataFrame([features])
    X = pd.get_dummies(X)
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
    try:
        path = nx.shortest_path(G, start_node, school_node, weight="weight")
        cost = nx.shortest_path_length(G, start_node, school_node, weight="weight")

        st.success("Best route found!")
        st.write(" → ".join(path))
        st.write(f"Total route cost: {cost:.4f}")

        fig = draw_city(city_grid, path)
        st.pyplot(fig)

    except nx.NetworkXNoPath:
        st.error("No valid route found.")
