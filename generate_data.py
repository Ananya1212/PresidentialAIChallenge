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
# Global parameters
# ----------------------------
NUM_NEIGHBORHOODS = 8
NUM_ROADS = 20
TIME_WINDOWS_PER_DAY = 48   # 30-minute intervals
DAYS = 5                   # Monday–Friday
TARGET_ROWS_APPROX = NUM_ROADS * TIME_WINDOWS_PER_DAY * DAYS  # ~4,800
# We'll double via neighborhood variation to reach ~9,600

# ----------------------------
# School schedule
# ----------------------------
SCHOOL_START = 8
SCHOOL_END = 15

# ----------------------------
# Neighborhood definitions
# ----------------------------
neighborhoods = []
for i in range(NUM_NEIGHBORHOODS):
    neighborhoods.append({
        "neighborhood_id": f"N{i}",
        "neighborhood_population": random.randint(3000, 9000),
        "working_population_pct": round(random.uniform(0.45, 0.7), 2),
        "students_population": random.randint(400, 1200)
    })

neighborhood_df = pd.DataFrame(neighborhoods)

# ----------------------------
# Road network
# ----------------------------
roads = []
for i in range(NUM_ROADS):
    roads.append({
        "road_id": f"R{i}",
        "start_node": random.choice(neighborhood_df["neighborhood_id"]),
        "end_node": random.choice(neighborhood_df["neighborhood_id"].tolist() + ["SCHOOL"]),
        "num_lanes": random.choice([1, 2, 3]),
        "speed_limit": random.choice([25, 30, 35, 40]),
        "distance_km": round(random.uniform(0.3, 3.0), 2),
        "is_intersection": random.choice([0, 1])
    })

roads_df = pd.DataFrame(roads)

# ----------------------------
# Weather generator
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
# Time-based congestion weight
# ----------------------------
def time_congestion_weight(hour):
    if 7 <= hour <= 9:
        return 1.8   # arrival peak
    elif 14 <= hour <= 16:
        return 1.7   # dismissal peak
    elif 6 <= hour <= 19:
        return 1.2   # normal daytime
    else:
        return 0.6   # late night / early morning

# ----------------------------
# Congestion level logic
# ----------------------------
def congestion_level(traffic_volume):
    if traffic_volume < 40:
        return "LOW"
    elif traffic_volume < 75:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# Accident risk logic
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
# Dataset generation
# ----------------------------
rows = []

for day in range(DAYS):
    for t in range(TIME_WINDOWS_PER_DAY):
        hour = t // 2  # 30-minute bins

        is_school_day = 1
        is_arrival_time = 1 if 7 <= hour <= 9 else 0
        is_dismissal_time = 1 if 14 <= hour <= 16 else 0

        weather, precipitation, visibility = generate_weather()
        time_weight = time_congestion_weight(hour)

        for _, road in roads_df.iterrows():
            # sample 2 neighborhoods per road-time to increase dataset size
            for _ in range(2):
                neighborhood = neighborhood_df.sample(1).iloc[0]

                base_volume = (
                    neighborhood["neighborhood_population"] *
                    neighborhood["working_population_pct"]
                )

                traffic_volume = int(
                    (base_volume / 120) * time_weight +
                    random.randint(5, 20)
                )

                avg_speed = max(
                    10,
                    road["speed_limit"] - traffic_volume * 0.25
                )

                crosswalk_present = random.choice([0, 1])
                crossing_guard_present = 1 if is_arrival_time or is_dismissal_time else 0

                congestion = congestion_level(traffic_volume)
                risk = accident_risk(
                    congestion,
                    precipitation,
                    road["is_intersection"],
                    crossing_guard_present
                )

                rows.append({
                    "hour": hour,
                    "day_of_week": day,
                    "is_school_day": is_school_day,
                    "is_arrival_time": is_arrival_time,
                    "is_dismissal_time": is_dismissal_time,
                    "weather_condition": weather,
                    "precipitation": precipitation,
                    "visibility_level": visibility,
                    "road_id": road["road_id"],
                    "start_node": road["start_node"],
                    "end_node": road["end_node"],
                    "num_lanes": road["num_lanes"],
                    "speed_limit": road["speed_limit"],
                    "distance_km": road["distance_km"],
                    "is_intersection": road["is_intersection"],
                    "neighborhood_id": neighborhood["neighborhood_id"],
                    "neighborhood_population": neighborhood["neighborhood_population"],
                    "working_population_pct": neighborhood["working_population_pct"],
                    "students_population": neighborhood["students_population"],
                    "distance_to_school_m": random.randint(100, 2000),
                    "crosswalk_present": crosswalk_present,
                    "crossing_guard_present": crossing_guard_present,
                    "traffic_volume": traffic_volume,
                    "average_speed": round(avg_speed, 1),
                    "congestion_level": congestion,
                    "accident_risk": risk
                })

# ----------------------------
# Final dataset
# ----------------------------
df = pd.DataFrame(rows)
df.to_csv("/Users/ananyamadduri/Documents/Presidential_Challenge/Data/safeflow_ai_simulated_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print(f"Total rows: {len(df)}")