import numpy as np
import pandas as pd
import random

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Load grid-derived structures
# ----------------------------
roads_df = pd.read_csv("roads_raw.csv")
neighborhood_df = pd.read_csv("neighborhoods.csv")

# ----------------------------
# Global parameters
# ----------------------------
TIME_WINDOWS_PER_DAY = 48
DAYS = 5

SCHOOL_X, SCHOOL_Y = 18, 18

# ----------------------------
# Weather generator (UNCHANGED)
# ----------------------------
def generate_weather():
    weather = random.choices(
        ["clear", "rain", "fog"],
        weights=[0.65, 0.25, 0.10]
    )[0]
    precipitation = 1 if weather == "rain" else 0
    visibility = "low" if weather in ["fog", "rain"] else "high"
    return weather, precipitation, visibility

# ----------------------------
# Time congestion weight (UNCHANGED)
# ----------------------------
def time_congestion_weight(hour):
    if 7 <= hour <= 9:
        return 1.8
    elif 14 <= hour <= 16:
        return 1.7
    elif 6 <= hour <= 19:
        return 1.2
    else:
        return 0.6

# ----------------------------
# Congestion level logic (UNCHANGED)
# ----------------------------
def congestion_level(traffic_volume):
    if traffic_volume < 40:
        return "LOW"
    elif traffic_volume < 75:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# Accident risk logic (UNCHANGED)
# ----------------------------
def accident_risk(congestion, precipitation, is_intersection, crossing_guard):
    score = 0
    score += {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[congestion]
    score += 2 if precipitation else 0
    score += 2 if is_intersection else 0
    score -= 1 if crossing_guard else 0

    if score <= 3:
        return "LOW"
    elif score <= 6:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# Distance to school (GRID-AWARE)
# ----------------------------
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# ----------------------------
# Dataset generation
# ----------------------------
rows = []

for day in range(DAYS):
    for t in range(TIME_WINDOWS_PER_DAY):
        hour = t // 2

        is_school_day = 1
        is_arrival_time = 1 if 7 <= hour <= 9 else 0
        is_dismissal_time = 1 if 14 <= hour <= 16 else 0

        weather, precipitation, visibility = generate_weather()
        time_weight = time_congestion_weight(hour)

        for _, road in roads_df.iterrows():

            # sample 2 neighborhoods per road-time (same as old)
            for _ in range(2):
                neighborhood = neighborhood_df.sample(1).iloc[0]

                base_volume = (
                    neighborhood["neighborhood_population"] *
                    neighborhood["working_population_pct"]
                )

                traffic_volume = int(
                    (base_volume / 150) * time_weight +
                    random.randint(5, 20)
                )

                avg_speed = max(
                    10,
                    random.choice([25, 30, 35, 40]) - traffic_volume * 0.25
                )

                crosswalk_present = random.choice([0, 1])
                crossing_guard_present = 1 if (is_arrival_time or is_dismissal_time) else 0

                congestion = congestion_level(traffic_volume)

                # intersections = where road segments meet
                is_intersection = random.choice([0, 1])

                risk = accident_risk(
                    congestion,
                    precipitation,
                    is_intersection,
                    crossing_guard_present
                )

                dist_to_school = manhattan_distance(
                    road["to_x"], road["to_y"],
                    SCHOOL_X, SCHOOL_Y
                ) * 100  # meters

                rows.append({
                    "hour": hour,
                    "day_of_week": day,
                    "is_school_day": is_school_day,
                    "is_arrival_time": is_arrival_time,
                    "is_dismissal_time": is_dismissal_time,
                    "weather_condition": weather,
                    "precipitation": precipitation,
                    "visibility_level": visibility,
                    "road_id": f"R{road['road_id']}",
                    "start_node": f"({road['from_x']},{road['from_y']})",
                    "end_node": f"({road['to_x']},{road['to_y']})",
                    "num_lanes": random.choice([1, 2, 3]),
                    "speed_limit": random.choice([25, 30, 35, 40]),
                    "distance_km": round(0.1, 2),
                    "is_intersection": is_intersection,
                    "neighborhood_id": neighborhood["neighborhood_id"],
                    "neighborhood_population": neighborhood["neighborhood_population"],
                    "working_population_pct": neighborhood["working_population_pct"],
                    "students_population": neighborhood["students_population"],
                    "distance_to_school_m": dist_to_school,
                    "crosswalk_present": crosswalk_present,
                    "crossing_guard_present": crossing_guard_present,
                    "traffic_volume": traffic_volume,
                    "average_speed": round(avg_speed, 1),
                    "congestion_level": congestion,
                    "accident_risk": risk
                })

# ----------------------------
# Save dataset
# ----------------------------
df = pd.DataFrame(rows)
df.to_csv("safeflow_ai_simulated_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print(f"Total rows: {len(df)}")
