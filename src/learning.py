import sys
import random
import math
import time
from sklearn.ensemble import RandomForestRegressor

sample_partitioning = []
time_projection = []
time_refinement = []
time_scan = []

sample_path = sys.argv[2]

with open(sample_path, "r") as f:
    for line in f:
        result = list(map(int, line.split()))
        if len(result) == 1:
            continue
        time_projection.append(result[-3])
        time_refinement.append(result[-2])
        time_scan.append(result[-1])
        sample_partitioning.append(result[:-3])

model_projection = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", bootstrap=False, random_state=42)
model_refinement = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", bootstrap=False, random_state=42)
model_scan = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", bootstrap=False, random_state=42)
model_projection.fit(sample_partitioning, time_projection)
model_refinement.fit(sample_partitioning, time_refinement)
model_scan.fit(sample_partitioning, time_scan)

dimension = len(sample_partitioning[0])
num_start = 1
best_partitioning = []
best_time = float("inf")

for sort_dimension in range(dimension):
    for i in range(num_start):
        candidate_partitioning = []
        for d in range(dimension):
            candidate_partitioning.append(random.randint(2, 100))
        candidate_partitioning[sort_dimension] = 1
        candidate_time = model_projection.predict([candidate_partitioning])[0] + model_refinement.predict([candidate_partitioning])[0] + model_scan.predict([candidate_partitioning])[0]

        start_temperature = candidate_time
        end_temperature = 1
        time_limit = float(sys.argv[1]) / dimension
        start_time = time.time()

        while time.time() - start_time < time_limit:
            new_candidate_partitioning = []
            for d in range(dimension):
                new_candidate_partitioning.append(min(max(candidate_partitioning[d] + random.randint(-5, 5), 2), 100))
            new_candidate_partitioning[sort_dimension] = 1
            new_candidate_time = model_projection.predict([new_candidate_partitioning])[0] + model_refinement.predict([new_candidate_partitioning])[0] + model_scan.predict([new_candidate_partitioning])[0]

            temperature = start_temperature + (end_temperature - start_temperature) * (time.time() - start_time) / time_limit
            diff = (candidate_time - new_candidate_time) / temperature
            if diff >= 0:
                probability = 1
            elif diff < -500:
                probability = 0
            else:
                probability = math.exp((candidate_time - new_candidate_time) / temperature)

            if probability > random.random():
                candidate_partitioning = new_candidate_partitioning
                candidate_time = new_candidate_time

        if candidate_time < best_time:
            best_partitioning = candidate_partitioning
            best_time = candidate_time

print(*best_partitioning)
