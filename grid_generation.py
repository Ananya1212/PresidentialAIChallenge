import random
import pandas as pd

GRID_SIZE = 20
random.seed(42)

BUILDING = "building"
ROAD = "road"
NEIGHBORHOOD = "neighborhood"
SCHOOL = "school"

grid = [[BUILDING for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

roads = []
road_id = 0

def add_horizontal_road(y, x_start, x_end, name):
    global road_id
    for x in range(x_start, x_end):
        grid[y][x] = ROAD
        grid[y][x + 1] = ROAD
        roads.append({
            "road_id": road_id,
            "street_name": name,
            "from_x": x,
            "from_y": y,
            "to_x": x + 1,
            "to_y": y
        })
        road_id += 1

def add_vertical_road(x, y_start, y_end, name):
    global road_id
    for y in range(y_start, y_end):
        grid[y][x] = ROAD
        grid[y + 1][x] = ROAD
        roads.append({
            "road_id": road_id,
            "street_name": name,
            "from_x": x,
            "from_y": y,
            "to_x": x,
            "to_y": y + 1
        })
        road_id += 1

# Roads (non-uniform)
add_horizontal_road(4, 1, 18, "Maple Ave")
add_horizontal_road(9, 3, 16, "Oak St")
add_horizontal_road(14, 0, 12, "Pine Blvd")

add_vertical_road(3, 2, 17, "1st St")
add_vertical_road(7, 0, 14, "2nd St")
add_vertical_road(12, 5, 19, "3rd St")
add_vertical_road(17, 1, 10, "4th St")

# School
grid[18][18] = SCHOOL
grid[18][17] = ROAD

# Neighborhoods
neighborhood_positions = [(2,2), (6,6), (10,3), (5,15), (14,8), (9,17), (16,5)]

neighborhoods = []
for i, (x, y) in enumerate(neighborhood_positions):
    grid[y][x] = NEIGHBORHOOD
    neighborhoods.append({
        "neighborhood_id": f"N{i}",
        "neighborhood_population": random.randint(3000, 9000),
        "working_population_pct": round(random.uniform(0.45, 0.7), 2),
        "students_population": random.randint(400, 1200),
        "x": x,
        "y": y
    })
    grid[y][x + 1] = ROAD

# Save files
pd.DataFrame(roads).to_csv("roads_raw.csv", index=False)
pd.DataFrame(neighborhoods).to_csv("neighborhoods.csv", index=False)

print("Grid generated!")
print("Files created: roads_raw.csv, neighborhoods.csv")
