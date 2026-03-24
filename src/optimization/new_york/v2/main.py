import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import EVConfig, PREDEFINED_ROUTES, SELECTED_ROUTE_INDEX
from data_processor import NewYorkDataProcessor
from route_planner import EVRoutePlanner
from optimizer import NSGAIIOptimizer
from visualizer import RouteVisualizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')


def route_out_dir(start, end):
    folder = os.path.join(OUT_DIR, f"{start}_{end}")
    os.makedirs(folder, exist_ok=True)
    return folder


def out(folder, filename):
    return os.path.join(folder, filename)


def astar_heuristic(graph):
    def h(u, v):
        lat1 = graph.nodes[u].get('lat', 0)
        lon1 = graph.nodes[u].get('lon', 0)
        lat2 = graph.nodes[v].get('lat', 0)
        lon2 = graph.nodes[v].get('lon', 0)
        dlat = abs(lat2 - lat1) * 111.0
        dlon = abs(lon2 - lon1) * 82.6
        return (dlat ** 2 + dlon ** 2) ** 0.5
    return h


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ev_config = EVConfig()

    print("Loading data...")
    processor = NewYorkDataProcessor('../data/')
    graph, stations, speed_data, availability_data, speed_ts, avail_ts = processor.load(use_cache=True)

    planner = EVRoutePlanner(graph, stations, speed_data, availability_data, ev_config, speed_ts, avail_ts)
    optimizer = NSGAIIOptimizer(planner)

    print(f"\nNetwork: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"EV Stations: {len(stations)}")

    route = PREDEFINED_ROUTES[SELECTED_ROUTE_INDEX]
    start, end = route["start"], route["end"]
    print(f"\nUsing predefined route {SELECTED_ROUTE_INDEX + 1} ({route['distance_km']:.2f} km): {start} -> {end}")

    run_dir = route_out_dir(start, end)
    print(f"Saving outputs to: {run_dir}")

    try:
        h = astar_heuristic(graph)
        astar_path = nx.astar_path(graph, start, end, heuristic=h, weight='distance_km')
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
          f"{objectives[best_dist_idx].total_time_min:.1f} min")
    print(f"Best by time     : {objectives[best_time_idx].distance_km:.2f} km, "
          f"{objectives[best_time_idx].total_time_min:.1f} min")

    visualizer = RouteVisualizer()

    pareto_fig = visualizer.plot_pareto_front(solutions, objectives,
                                              title="New York EV - Pareto Front")
    plt.savefig(out(run_dir, 'pareto_front.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pareto_front.png")

    fig_3d = RouteVisualizer.plot_3d_pareto_front(
        solutions, objectives, title="NSGA-II Optimisation",
        start_node=start, end_node=end
    )
    plt.savefig(out(run_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved pareto_front_3d.png")

    para_fig = RouteVisualizer.plot_parallel_coordinates(
        objectives, title="New York EV Route Objectives"
    )
    para_fig.write_image(out(run_dir, 'parallel_coordinates.png'))
    para_fig.show()
    print("Saved parallel_coordinates.png")

    detail_fig = visualizer.plot_objectives_detail(objectives)
    plt.savefig(out(run_dir, 'objectives_detail.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved objectives_detail.png")

    selected = [solutions[best_dist_idx], solutions[best_time_idx]]
    selected_obj = [objectives[best_dist_idx], objectives[best_time_idx]]
    labels = ['Shortest Distance', 'Fastest Time']

    if astar_path:
        selected.append(astar_path)
        selected_obj.append(astar_obj)
        labels.append('A* Baseline')

    route_fig = visualizer.plot_route_comparison(
        graph, selected, selected_obj, stations,
        route_planner=planner, labels=labels
    )
    route_fig.write_html(out(run_dir, 'route_comparison.html'))
    route_fig.show()
    print("Saved route_comparison.html")

    RouteVisualizer.plot_route_networkx(
        graph, selected, start, end,
        labels=labels, route_planner=planner,
        output_path=out(run_dir, 'route_networkx.png')
    )
    plt.show()
    print("Saved route_networkx.png")

    net_fig = RouteVisualizer.plot_network(
        graph, stations, title="New York Road Network & EV Stations"
    )
    net_fig.write_html(out(OUT_DIR, 'network.html'))
    net_fig.show()
    print("Saved network.html")

    print()
    for path, obj, label in zip(selected, selected_obj, labels):
        if path:
            summary = RouteVisualizer.create_route_summary(
                graph, path, obj, planner, route_name=label
            )
            print(summary)
            safe_label = label.replace(' ', '_').replace('*', 'astar')
            with open(out(run_dir, f'summary_{safe_label}.txt'), 'w') as f:
                f.write(summary)

    print("\nDone!")
    return solutions, objectives


if __name__ == '__main__':
    result = main()
