import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import EVConfig
from data_processor import ChicagoDataProcessor
from route_planner import RoutePlanner
from optimizer import NSGAIIOptimizer
from visualizer import Visualizer


def find_route_endpoints(graph, min_distance=40.0, max_attempts=5000):
    nodes = list(graph.nodes())

    for attempt in range(max_attempts):
        start = random.choice(nodes)
        end = random.choice(nodes)

        if start == end:
            continue

        try:
            dist = nx.shortest_path_length(graph, start, end, weight='distance_km')
            if dist > min_distance:
                print(f"Found endpoints (attempt {attempt + 1}): {dist:.1f} km apart")
                return start, end
        except nx.NetworkXNoPath:
            continue

    return None, None


def main():
    ev_config = EVConfig()

    print("Loading data...")
    processor = ChicagoDataProcessor('data/')
    graph, stations, speed_data, availability_data = processor.load()

    planner = RoutePlanner(graph, stations, speed_data, availability_data, ev_config)
    optimizer = NSGAIIOptimizer(planner)

    print(f"\nNetwork: {graph.number_of_nodes()} nodes")

    start, end = find_route_endpoints(graph)
    if not start:
        print("Could not find valid endpoints!")
        return None

    print(f"Start: {start}, End: {end}")

    solutions, objectives = optimizer.optimize(start, end, pop_size=100, generations=50)

    if not solutions:
        print("No solutions found!")
        return None

    print(f"\nFound {len(solutions)} Pareto-optimal solutions")

    best_dist_idx = np.argmin([o.distance_km for o in objectives])
    best_time_idx = np.argmin([o.total_time_min for o in objectives])

    print(f"\nBest by distance: {objectives[best_dist_idx].distance_km:.1f} km, "
          f"{objectives[best_dist_idx].total_time_min:.1f} min")
    print(f"Best by time: {objectives[best_time_idx].distance_km:.1f} km, "
          f"{objectives[best_time_idx].total_time_min:.1f} min")

    visualizer = Visualizer()

    pareto_fig = visualizer.plot_pareto_front(solutions, objectives)
    plt.savefig('outputs/pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()

    selected = [solutions[best_dist_idx], solutions[best_time_idx]]
    selected_obj = [objectives[best_dist_idx], objectives[best_time_idx]]

    route_fig = visualizer.plot_route_comparison(
        graph, selected, selected_obj, stations,
        route_planner=planner,
        labels=['Shortest', 'Fastest']
    )
    route_fig.write_html('outputs/route_comparison.html')
    route_fig.show()

    print("\nDone!")
    return solutions, objectives


if __name__ == '__main__':
    result = main()
