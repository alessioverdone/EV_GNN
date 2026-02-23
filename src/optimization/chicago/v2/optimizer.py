import random
import copy
import numpy as np
import networkx as nx
from typing import List, Tuple
from enum import Enum

from config import PathObjectives
from route_planner import EVRoutePlanner


class CrossoverType(Enum):
    NBX = 0  # Node-based crossover (single common node)
    MNX = 1  # Multiple node crossover
    PMX = 2  # Partially mapped crossover


class MutationType(Enum):
    NONE = 0
    ENABLED = 1


class NSGAIIOptimizer:

    def __init__(self, planner: EVRoutePlanner,
                 crossover_type: CrossoverType = CrossoverType.NBX,
                 mutation_type: MutationType = MutationType.ENABLED,
                 mutation_probability: float = 0.3,
                 tournament_size: int = 2):
        self.planner = planner
        self.graph = planner.graph
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size

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

    def _generate_random_path(self, start, end) -> List:
        path = []
        current = start

        while current != end:
            path.append(current)
            neighbors = list(self.graph.successors(current))

            if not neighbors:
                return path

            unvisited = [n for n in neighbors if n not in path]

            if not unvisited:
                return path

            current = random.choice(unvisited)

        path.append(end)
        return path

    def _generate_valid_random_path(self, start, end, max_attempts: int = 500) -> List:
        for _ in range(max_attempts):
            path = self._generate_random_path(start, end)
            if path and path[0] == start and path[-1] == end:
                return path
        return []

    def create_population(self, start, end, size: int) -> List[List]:
        print(f"\nCreating population: {size} unique random paths")
        population = []
        seen = set()

        for _ in range(size * 5):
            if len(population) >= size:
                break

            path = self._generate_valid_random_path(start, end)
            if not path:
                continue

            key = tuple(path)
            if key not in seen:
                seen.add(key)
                population.append(path)

        print(f"Generated: {len(population)} unique paths")
        return population

    def _tournament_selection(self, population: List, objectives: List[PathObjectives]) -> List:
        candidates = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        winner = candidates[0]

        for candidate in candidates[1:]:
            if self._dominates(objectives[candidate].as_array(), objectives[winner].as_array()):
                winner = candidate

        return copy.copy(population[winner])

    def _crossover(self, p1: List, p2: List) -> List:
        if self.crossover_type == CrossoverType.NBX:
            return self._nbx_crossover(p1, p2)
        elif self.crossover_type == CrossoverType.MNX:
            return self._mnx_crossover(p1, p2)
        elif self.crossover_type == CrossoverType.PMX:
            return self._pmx_crossover(p1, p2)
        return copy.copy(p1)

    def _nbx_crossover(self, p1: List, p2: List) -> List:
        common = set(p1) & set(p2)
        common.discard(p1[0])
        common.discard(p1[-1])

        if not common:
            return copy.copy(p1) if random.random() < 0.5 else copy.copy(p2)

        node = random.choice(list(common))
        offspring = p1[:p1.index(node)] + p2[p2.index(node):]
        offspring = self._remove_loops(offspring)

        return offspring if self.is_valid(offspring) else (copy.copy(p1) if random.random() < 0.5 else copy.copy(p2))

    def _mnx_crossover(self, p1: List, p2: List) -> List:
        common = list(set(p1) & set(p2) - {p1[0], p1[-1]})

        if not common:
            return self._nbx_crossover(p1, p2)

        num_points = random.randint(1, min(3, len(common)))
        points = sorted(random.sample(common, num_points), key=lambda n: p1.index(n))

        offspring = list(p1)
        take_from_p2 = False

        for point in points:
            if point in offspring and point in p2:
                idx = offspring.index(point)
                p2_idx = p2.index(point)
                if take_from_p2:
                    offspring = offspring[:idx] + p2[p2_idx:]
                take_from_p2 = not take_from_p2

        offspring = self._remove_loops(offspring)
        return offspring if self.is_valid(offspring) else (copy.copy(p1) if random.random() < 0.5 else copy.copy(p2))

    def _pmx_crossover(self, p1: List, p2: List) -> List:
        if len(p1) < 3:
            return copy.copy(p1)

        i = random.randint(1, len(p1) - 2)
        j = random.randint(i, min(i + 5, len(p1) - 1))

        segment = p1[i:j]
        segment_set = set(segment)

        prefix = [n for n in p2 if n not in segment_set]
        insert_at = min(i, len(prefix))
        offspring = prefix[:insert_at] + segment + prefix[insert_at:]

        if not offspring or offspring[0] != p1[0] or offspring[-1] != p1[-1]:
            return copy.copy(p1)

        offspring = self._remove_loops(offspring)
        return offspring if self.is_valid(offspring) else copy.copy(p1)

    def _mutate(self, path: List) -> List:
        if self.mutation_type == MutationType.NONE:
            return path

        if random.random() > self.mutation_probability or len(path) < 3:
            return path

        i = random.randint(1, len(path) - 2)
        j = random.randint(i + 1, min(i + 5, len(path) - 1))

        segment = self._generate_valid_random_path(path[i - 1], path[j], max_attempts=50)

        if segment:
            mutated = path[:i] + segment[1:-1] + path[j:]
            mutated = self._remove_loops(mutated)
            if self.is_valid(mutated):
                return mutated

        return path

    def _remove_loops(self, path: List) -> List:
        cleaned = []
        seen = {}

        for node in path:
            if node in seen:
                cleaned = cleaned[:seen[node] + 1]
                seen = {n: idx for idx, n in enumerate(cleaned)}
            else:
                seen[node] = len(cleaned)
                cleaned.append(node)

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

            obj_range = (objectives[sorted_idx[-1]].as_array()[obj_idx] -
                         objectives[sorted_idx[0]].as_array()[obj_idx])

            if obj_range > 0:
                for i in range(1, len(sorted_idx) - 1):
                    idx = front.index(sorted_idx[i])
                    diff = (objectives[sorted_idx[i + 1]].as_array()[obj_idx] -
                            objectives[sorted_idx[i - 1]].as_array()[obj_idx])
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

        objectives = [self.planner.evaluate(p) for p in population]

        for gen in range(generations):
            offspring = []

            for _ in range(pop_size):
                p1 = self._tournament_selection(population, objectives)
                p2 = self._tournament_selection(population, objectives)
                child = self._crossover(p1, p2)
                child = self._mutate(child)

                if child and self.is_valid(child):
                    offspring.append(child)

            if not offspring:
                offspring = [copy.copy(random.choice(population)) for _ in range(pop_size // 2)]

            offspring_obj = [self.planner.evaluate(p) for p in offspring]

            combined = population + offspring
            combined_obj = objectives + offspring_obj

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
