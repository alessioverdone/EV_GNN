import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import EVConfig
from data_processor import DenmarkDataProcessor
from route_planner import EVRoutePlanner
from optimizer import NSGAIIOptimizer
from visualizer import RouteVisualizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
TRAFFIC_DIR = os.path.join(SCRIPT_DIR, '/Users/denmark/data/traffic_feb_june')
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')


def route_out_dir(start, end):
    folder = os.path.join(OUT_DIR, f"{start}_{end}")
    os.makedirs(folder, exist_ok=True)
    return folder


def out(folder, filename):
    return os.path.join(folder, filename)


def find_route_endpoints(graph, start_hint='4320', end_hint='4551',
                         min_distance_km: float = 0.5,
                         max_attempts: int = 3000):
    nodes = list(graph.nodes())

    if start_hint in graph and end_hint in graph:
        try:
            dist = nx.shortest_path_length(graph, start_hint, end_hint, weight='distance_km')
            if dist >= min_distance_km:
                print(f"Using nodes {start_hint} -> {end_hint}: {dist:.2f} km apart")
                return start_hint, end_hint
        except nx.NetworkXNoPath:
            pass

    for attempt in range(max_attempts):
        start = random.choice(nodes)
        end = random.choice(nodes)

        if start == end:
            continue

        try:
            dist = nx.shortest_path_length(graph, start, end, weight='distance_km')
            if dist >= min_distance_km:
                print(f"Found endpoints (attempt {attempt + 1}): {dist:.2f} km apart")
                return start, end
        except nx.NetworkXNoPath:
            continue

    return None, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ev_config = EVConfig()

    print("Loading Denmark data...")
    processor = DenmarkDataProcessor(data_path=DATA_DIR, traffic_path=TRAFFIC_DIR)
    graph, stations, speed_data, availability_data, speed_ts, avail_ts = processor.load()

    planner = EVRoutePlanner(graph, stations, speed_data, availability_data, ev_config, speed_ts, avail_ts)
    optimizer = NSGAIIOptimizer(planner)

    print(f"\nNetwork: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"EV Stations: {len(stations)}")

    start, end = find_route_endpoints(graph)
    if not start:
        print("Could not find valid endpoints!")
        return None

    print(f"\nRoute: {start} -> {end}")

    run_dir = route_out_dir(start, end)
    print(f"Saving outputs to: {run_dir}")

    try:
        astar_path = nx.astar_path(graph, start, end, weight='distance_km')
        astar_obj = planner.evaluate(astar_path)
        print(f"\nA* baseline:")
        print(f"  Distance  : {astar_obj.distance_km:.2f} km")
        print(f"  Travel    : {astar_obj.travel_time_min:.1f} min")
        print(f"  Charging  : {astar_obj.charging_time_min:.1f} min")
        print(f"  Stops     : {astar_obj.num_charging_stops}")
        print(f"  Final SOC : {astar_obj.final_soc * 100:.1f}%")
    except nx.NetworkXNoPath:
        astar_path = None
        astar_obj = None
        print("A* path not found")

    solutions, objectives = optimizer.optimize(start, end, pop_size=100, generations=50)
    if not solutions:
        print("No solutions found!")
        return None

    print(f"\nFound {len(solutions)} Pareto-optimal solutions")

    best_dist_idx = int(np.argmin([o.distance_km for o in objectives]))
    best_time_idx = int(np.argmin([o.total_time_min for o in objectives]))

    print(f"\nBest by distance : {objectives[best_dist_idx].distance_km:.2f} km, "
          f"{objectives[best_dist_idx].total_time_min:.1f} min total")
    print(f"Best by time     : {objectives[best_time_idx].distance_km:.2f} km, "
          f"{objectives[best_time_idx].total_time_min:.1f} min total")

    visualizer = RouteVisualizer()

    pareto_fig = visualizer.plot_pareto_front(solutions, objectives, title="Denmark EV - Pareto Front")
    plt.savefig(out(run_dir, 'pareto_front.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pareto_front.png")

    fig_3d = RouteVisualizer.plot_3d_pareto_front(
        solutions, objectives, title="NSGA-II Optimisation", start_node=start, end_node=end
    )
    plt.savefig(out(run_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pareto_front_3d.png")

    para_fig = RouteVisualizer.plot_parallel_coordinates(objectives, title="Denmark EV Route Objectives")
    para_fig.write_image(out(run_dir, 'parallel_coordinates.png'))
    para_fig.show()
    print(f"Saved parallel_coordinates.png")

    selected = [solutions[best_dist_idx], solutions[best_time_idx]]
    selected_obj = [objectives[best_dist_idx], objectives[best_time_idx]]
    labels = ['Shortest Distance', 'Fastest Time']

    if astar_path:
        selected.append(astar_path)
        selected_obj.append(astar_obj)
        labels.append('A* Baseline')

    route_fig = visualizer.plot_route_comparison(
        graph, selected, selected_obj, stations, route_planner=planner, labels=labels
    )
    route_fig.write_html(out(run_dir, 'route_comparison.html'))
    route_fig.show()
    print(f"Saved route_comparison.html")

    RouteVisualizer.plot_route_networkx(
        graph, selected, start, end, labels=labels, route_planner=planner,
        output_path=out(run_dir, 'route_networkx.png')
    )
    plt.show()
    print(f"Saved route_networkx.png")

    net_fig = RouteVisualizer.plot_network(graph, stations, title="Denmark Road Network & EV Stations")
    net_fig.write_html(out(OUT_DIR, 'network.html'))
    net_fig.show()
    print(f"Saved network.html")

    print("\nDone!")
    return solutions, objectives


if __name__ == '__main__':
    main()
