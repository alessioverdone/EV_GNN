import random
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from geopy.distance import geodesic

from config import PathObjectives
from route_planner import EVRoutePlanner


class NSGAIIOptimizer:

    def __init__(self, planner: EVRoutePlanner):
        self.planner = planner
        self.graph = planner.graph

    def is_valid(self, path: List) -> bool:
        if not path or len(path) < 2:
            return False

        for node in path:
            if node not in self.graph:
                return False

        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False

        return True

    def create_population(self, start, end, size: int) -> List[List]:
        random_target = size // 2
        biased_target = size - random_target

        print(f"\nCreating population: {size} paths ({random_target} random, {biased_target} biased)")

        biased = self._generate_biased_paths(start, end, biased_target)
        random_paths = self._generate_random_paths(start, end, random_target, biased)

        population = biased + random_paths
        print(f"Generated: {len(biased)} biased + {len(random_paths)} random = {len(population)} paths")

        return population

    def _generate_biased_paths(self, start, end, target: int) -> List[List]:
        paths = []

        try:
            shortest = nx.shortest_path(self.graph, start, end, weight='distance_km')
            paths.append(shortest)
            dist = sum(self.graph.edges[shortest[i], shortest[i + 1]]['distance_km']
                       for i in range(len(shortest) - 1))
            print(f"  Shortest path: {len(shortest)} nodes, {dist:.1f} km")
        except nx.NetworkXNoPath:
            print(f"  No path between {start} and {end}")
            return []

        paths.extend(self._speed_optimized_paths(start, end, paths))
        paths.extend(self._availability_paths(start, end, target, paths))
        paths.extend(self._multi_stop_paths(start, end, target, paths))

        print(f"  Biased paths: {len(paths)}")
        return paths

    def _speed_optimized_paths(self, start, end, existing: List) -> List[List]:
        paths = []

        def speed_weight(u, v, d):
            dist = d.get('distance_km', 1.0)
            site_id = d.get('site_id', '')
            speed = 30.0
            if site_id in self.planner.traffic_data:
                speeds = list(self.planner.traffic_data[site_id].values())
                if speeds:
                    speed = max(np.mean(speeds), 5.0)
            return dist / speed

        try:
            fastest = nx.shortest_path(self.graph, start, end, weight=speed_weight)
            if fastest not in existing:
                paths.append(fastest)
        except:
            pass

        return paths

    def _get_avg_availability(self, site_id: str) -> float:
        if site_id in self.planner.availability_data:
            values = list(self.planner.availability_data[site_id].values())
            if values:
                return np.mean(values)
        return 0.0

    def _availability_paths(self, start, end, target: int, existing: List) -> List[List]:
        paths = []

        if not self.planner.ev_stations:
            return paths

        stations = [(sid, s, self._get_avg_availability(sid))
                    for sid, s in self.planner.ev_stations.items()]
        stations.sort(key=lambda x: x[2], reverse=True)

        for sid, station, _ in stations[:50]:
            if len(existing) + len(paths) >= target:
                break

            for node in [station.src_node, station.tgt_node]:
                if node and node in self.graph and node != start and node != end:
                    try:
                        p1 = nx.shortest_path(self.graph, start, node, weight='distance_km')
                        p2 = nx.shortest_path(self.graph, node, end, weight='distance_km')
                        path = p1 + p2[1:]

                        if not self._is_similar(path, existing + paths):
                            paths.append(path)
                            break
                    except:
                        continue

        return paths

    def _multi_stop_paths(self, start, end, target: int, existing: List) -> List[List]:
        paths = []

        if not self.planner.ev_stations:
            return paths

        charging_nodes = list(set(
            s.src_node for s in self.planner.ev_stations.values()
            if s.src_node and s.src_node in self.graph
        ))

        for _ in range(min(30, target - len(existing))):
            if len(charging_nodes) < 2:
                break

            waypoints = random.sample(charging_nodes, random.randint(1, 2))

            try:
                path = []
                current = start

                for wp in waypoints:
                    if wp != current and wp != end:
                        segment = nx.shortest_path(self.graph, current, wp, weight='distance_km')
                        path = segment if not path else path + segment[1:]
                        current = wp

                final = nx.shortest_path(self.graph, current, end, weight='distance_km')
                path = path + final[1:] if path else final

                if not self._is_similar(path, existing + paths):
                    paths.append(path)
            except:
                continue

        return paths

    def _generate_random_paths(self, start, end, target: int, existing: List) -> List[List]:
        paths = []
        all_nodes = list(self.graph.nodes())

        for _ in range(target * 5):
            if len(paths) >= target // 2:
                break

            randomness = random.choice([0.3, 0.5, 0.7, 0.9])
            path = self._random_walk(start, end, randomness)

            if path and self.is_valid(path) and not self._is_similar(path, existing + paths):
                paths.append(path)

        for _ in range(target * 10):
            if len(paths) >= target:
                break

            intermediates = random.sample(
                [n for n in all_nodes if n != start and n != end],
                min(random.randint(1, 3), len(all_nodes) - 2)
            )

            try:
                path = []
                current = start

                for inter in intermediates:
                    segment = nx.shortest_path(self.graph, current, inter, weight='distance_km')
                    path = segment if not path else path + segment[1:]
                    current = inter

                final = nx.shortest_path(self.graph, current, end, weight='distance_km')
                path = path + final[1:]

                if self.is_valid(path) and not self._is_similar(path, existing + paths):
                    paths.append(path)
            except:
                continue

        return paths

    def _random_walk(self, start, end, randomness: float = 0.5) -> List:
        visited = set()
        path = [start]
        current = start

        for _ in range(1000):
            if current == end:
                break

            neighbors = [n for n in self.graph.successors(current) if n not in visited]

            if not neighbors:
                if len(path) > 1:
                    path.pop()
                    visited.discard(current)
                    current = path[-1]
                else:
                    break
            else:
                if random.random() < randomness:
                    next_node = random.choice(neighbors)
                else:
                    next_node = self._nearest_to_target(neighbors, end)

                path.append(next_node)
                visited.add(next_node)
                current = next_node

        if current != end:
            try:
                tail = nx.shortest_path(self.graph, current, end, weight='distance_km')
                path = path + tail[1:]
            except:
                return []

        return path

    def _nearest_to_target(self, candidates: List, target) -> any:
        if target in candidates:
            return target

        target_data = self.graph.nodes[target]
        target_lat, target_lon = target_data.get('lat'), target_data.get('lon')

        if target_lat is None:
            return candidates[0] if candidates else None

        distances = []
        for node in candidates:
            node_data = self.graph.nodes[node]
            lat, lon = node_data.get('lat'), node_data.get('lon')

            if lat is not None:
                try:
                    dist = geodesic((lat, lon), (target_lat, target_lon)).kilometers
                    distances.append((node, dist))
                except:
                    distances.append((node, float('inf')))
            else:
                distances.append((node, float('inf')))

        valid = [(n, d) for n, d in distances if d != float('inf')]
        if not valid:
            return random.choice(candidates)

        valid.sort(key=lambda x: x[1])
        weights = [1.0 / (i + 1) for i in range(len(valid))]
        weights = [w / sum(weights) for w in weights]

        return np.random.choice([n for n, _ in valid], p=weights)

    def _is_similar(self, path: List, population: List[List], threshold: float = 0.7) -> bool:
        if not population:
            return False

        path_set = set(path)

        for existing in population:
            if path == existing:
                return True

            existing_set = set(existing)
            if not path_set or not existing_set:
                continue

            similarity = len(path_set & existing_set) / len(path_set | existing_set)
            if similarity > threshold:
                return True

        return False

    def _repair(self, path: List) -> List:
        if len(path) < 2:
            return path

        repaired = [path[0]]

        for i in range(1, len(path)):
            current = repaired[-1]
            next_node = path[i]

            if self.graph.has_edge(current, next_node):
                repaired.append(next_node)
            else:
                try:
                    segment = nx.shortest_path(self.graph, current, next_node, weight='distance_km')
                    repaired.extend(segment[1:])
                except:
                    continue

        return repaired

    def _crossover(self, p1: List, p2: List) -> List:
        if not p1 or not p2:
            return p1 or p2

        common = set(p1) & set(p2)
        common.discard(p1[0])
        common.discard(p1[-1])

        if not common:
            return p1 if random.random() < 0.5 else p2

        node = random.choice(list(common))
        offspring = p1[:p1.index(node)] + p2[p2.index(node):]
        offspring = self._remove_loops(offspring)
        offspring = self._repair(offspring)

        return offspring if self.is_valid(offspring) else (p1 if random.random() < 0.5 else p2)

    def _mutate(self, path: List, rate: float = 0.1) -> List:
        if random.random() > rate or len(path) < 3:
            return path

        i = random.randint(1, len(path) - 2)
        j = min(i + random.randint(1, 3), len(path) - 1)

        try:
            alt = nx.shortest_path(self.graph, path[i - 1], path[j], weight='distance_km')
            mutated = path[:i] + alt[1:-1] + path[j:]
            mutated = self._remove_loops(mutated)
            mutated = self._repair(mutated)
            return mutated if self.is_valid(mutated) else path
        except:
            return path

    def _remove_loops(self, path: List) -> List:
        if not path:
            return path

        cleaned = []
        seen = {}

        for node in path:
            if node in seen:
                cleaned = cleaned[:seen[node] + 1]
            else:
                cleaned.append(node)
                seen[node] = len(cleaned) - 1

        return cleaned

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _non_dominated_sort(self, population: List, objectives: List[PathObjectives]) -> List[List[int]]:
        n = len(population)
        dom_count = [0] * n
        dominated = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                obj_i, obj_j = objectives[i].as_array(), objectives[j].as_array()

                if self._dominates(obj_i, obj_j):
                    dominated[i].append(j)
                    dom_count[j] += 1
                elif self._dominates(obj_j, obj_i):
                    dominated[j].append(i)
                    dom_count[i] += 1

        for i in range(n):
            if dom_count[i] == 0:
                fronts[0].append(i)

        idx = 0
        while fronts[idx]:
            next_front = []
            for i in fronts[idx]:
                for j in dominated[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)

            idx += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts[:-1] if not fronts[-1] else fronts

    def _crowding_distance(self, front: List[int], objectives: List[PathObjectives]) -> List[float]:
        if len(front) <= 2:
            return [float('inf')] * len(front)

        distances = [0.0] * len(front)

        for obj_idx in range(2):
            sorted_idx = sorted(front, key=lambda x: objectives[x].as_array()[obj_idx])

            distances[front.index(sorted_idx[0])] = float('inf')
            distances[front.index(sorted_idx[-1])] = float('inf')

            obj_range = objectives[sorted_idx[-1]].as_array()[obj_idx] - objectives[sorted_idx[0]].as_array()[obj_idx]

            if obj_range > 0:
                for i in range(1, len(sorted_idx) - 1):
                    idx = front.index(sorted_idx[i])
                    diff = objectives[sorted_idx[i + 1]].as_array()[obj_idx] - objectives[sorted_idx[i - 1]].as_array()[obj_idx]
                    distances[idx] += diff / obj_range

        return distances

    def optimize(self, start, end, pop_size: int = 50, generations: int = 100) -> Tuple[List[List], List[PathObjectives]]:
        print(f"\nOptimizing: {start} -> {end}")

        population = self.create_population(start, end, pop_size)
        population = [p for p in population if self.is_valid(p)]

        if not population:
            print("No valid paths!")
            return [], []

        print(f"Starting with {len(population)} valid paths")

        for gen in range(generations):
            objectives = [self.planner.evaluate(p) for p in population]

            offspring = []
            for _ in range(pop_size * 3):
                if len(offspring) >= pop_size:
                    break

                p1, p2 = random.sample(population, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)

                if child and self.is_valid(child):
                    offspring.append(child)

            while len(offspring) < pop_size:
                offspring.append(random.choice(population).copy())

            combined = population + offspring
            combined_obj = objectives + [self.planner.evaluate(p) for p in offspring]

            fronts = self._non_dominated_sort(combined, combined_obj)

            new_pop, new_obj = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    for i in front:
                        new_pop.append(combined[i])
                        new_obj.append(combined_obj[i])
                else:
                    remaining = pop_size - len(new_pop)
                    distances = self._crowding_distance(front, combined_obj)
                    sorted_idx = sorted(range(len(front)), key=lambda x: distances[x], reverse=True)

                    for i in sorted_idx[:remaining]:
                        new_pop.append(combined[front[i]])
                        new_obj.append(combined_obj[front[i]])
                    break

            population, objectives = new_pop, new_obj

            if gen % 10 == 0:
                valid = [o for o in objectives if o.travel_time_min != float('inf')]
                if valid:
                    best = min(valid, key=lambda x: x.total_time_min)
                    print(f"Gen {gen}: time={best.total_time_min:.1f}min, dist={best.distance_km:.1f}km, stops={best.num_charging_stops}")

        fronts = self._non_dominated_sort(population, objectives)
        if fronts:
            pareto_idx = fronts[0]
            solutions = [population[i] for i in pareto_idx]
            objs = [objectives[i] for i in pareto_idx]

            valid = [(s, o) for s, o in zip(solutions, objs) if o.travel_time_min != float('inf')]
            if valid:
                return [s for s, _ in valid], [o for _, o in valid]

        return [], []
