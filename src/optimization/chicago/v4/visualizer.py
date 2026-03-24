
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from config import PathObjectives


class RouteVisualizer:

    @staticmethod
    def _get_avg_speed(route_planner, site_id: str) -> float:
        if route_planner and hasattr(route_planner, 'traffic_data'):
            if site_id in route_planner.traffic_data:
                values = list(route_planner.traffic_data[site_id].values())
                if values:
                    return np.mean(values)
        return 30.0

    @staticmethod
    def _get_avg_availability(route_planner, site_id: str) -> float:
        if route_planner and hasattr(route_planner, 'availability_data'):
            if site_id in route_planner.availability_data:
                values = list(route_planner.availability_data[site_id].values())
                if values:
                    return np.mean(values)
        return 5.0

    @staticmethod
    def plot_pareto_front(solutions: List[List[str]],
                          objectives: List[PathObjectives],
                          title: str = "Pareto Front"):
        fig = plt.figure(figsize=(15, 10))

        distances = [obj.distance_km for obj in objectives]
        total_times = [obj.total_time_min for obj in objectives]

        ax = fig.add_subplot(111)

        ax.scatter(distances, total_times, c='b', s=100, alpha=0.7, edgecolors='k')

        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Travel Time (min)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_route_comparison(graph: nx.DiGraph,
                              solutions: List[List[str]],
                              objectives: List[PathObjectives],
                              ev_stations: Dict,
                              route_planner=None,
                              labels: List[str] = None):
        fig = go.Figure()

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        used_charging_nodes = set()

        for idx, (path, obj) in enumerate(zip(solutions[:5], objectives[:5])):
            if not path:
                continue

            lats = []
            lons = []
            hover_texts = []

            if route_planner:
                route_charging_stops = route_planner.get_charging_stops_in_path(path)
                used_charging_nodes.update(route_charging_stops)
            else:
                route_charging_stops = []

            for node in path:
                node_data = graph.nodes[node]
                lats.append(node_data['lat'])
                lons.append(node_data['lon'])

                if node in route_charging_stops:
                    hover_texts.append(f"⚡ CHARGING STOP<br>Node: {node}")
                else:
                    hover_texts.append(f"Node: {node}")

            label = labels[idx] if labels else f"Route {idx + 1}"

            fig.add_trace(go.Scattermapbox(
                mode='markers+lines',
                lon=lons,
                lat=lats,
                marker={'size': 8, 'color': colors[idx % len(colors)]},
                text=hover_texts,
                name=f"{label}<br>Dist: {obj.distance_km:.1f}km<br>"
                     f"Time: {obj.travel_time_min + obj.charging_time_min:.1f}min<br>"
                     f"Charging stops: {obj.num_charging_stops}",
                line={'width': 3}
            ))

            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=[lons[0]],
                lat=[lats[0]],
                marker={'size': 15, 'color': 'green', 'symbol': 'circle'},
                text=[f"{label} - START"],
                name=f"{label} Start",
                showlegend=False
            ))

            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=[lons[-1]],
                lat=[lats[-1]],
                marker={'size': 15, 'color': 'red', 'symbol': 'square'},
                text=[f"{label} - END"],
                name=f"{label} End",
                showlegend=False
            ))

        used_station_lats = []
        used_station_lons = []
        used_station_names = []
        used_station_labels = []

        for i, node in enumerate(used_charging_nodes, 1):
            if node in graph.nodes:
                node_data = graph.nodes[node]
                used_station_lats.append(node_data['lat'])
                used_station_lons.append(node_data['lon'])
                used_station_labels.append(f"Stop {i}")

                station_info = "USED CHARGING STATION"
                if route_planner:
                    stations = route_planner.get_stations_at_node(node)
                    if stations:
                        station = stations[0]
                        avg_avail = RouteVisualizer._get_avg_availability(route_planner, station.station_id)
                        avg_speed = RouteVisualizer._get_avg_speed(route_planner, station.station_id)
                        station_info = (f"USED CHARGING STATION<br>"
                                        f"{station.name}<br>"
                                        f"ID: {station.station_id}<br>"
                                        f"Power: {station.charging_power_kw} kW<br>"
                                        f"Avg Availability: {avg_avail:.1f} chargers<br>"
                                        f"Avg Speed: {avg_speed:.1f} mph")

                used_station_names.append(station_info)

        if used_station_lats:
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=used_station_lons,
                lat=used_station_lats,
                marker={'size': 30, 'color': '#000000', 'symbol': 'circle', 'allowoverlap': True},
                hoverinfo='skip',
                showlegend=False
            ))
            fig.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=used_station_lons,
                lat=used_station_lats,
                marker={'size': 24, 'color': '#FFD700', 'symbol': 'circle', 'allowoverlap': True},
                text=used_station_labels,
                textposition='top right',
                textfont={'size': 13, 'color': '#FFD700'},
                customdata=used_station_names,
                hovertemplate='%{customdata}<extra></extra>',
                name=f'Used Charging Stations ({len(used_station_lats)})',
                showlegend=True
            ))

        other_station_lats = []
        other_station_lons = []
        other_station_names = []

        stations_to_show = min(100, len(ev_stations))
        for station in list(ev_stations.values())[:stations_to_show]:
            if station.src_node and station.src_node in graph:
                if station.src_node in used_charging_nodes:
                    continue

                node_data = graph.nodes[station.src_node]
                other_station_lats.append(node_data['lat'])
                other_station_lons.append(node_data['lon'])
                avg_avail = RouteVisualizer._get_avg_availability(route_planner, station.station_id)
                avg_speed = RouteVisualizer._get_avg_speed(route_planner, station.station_id)
                other_station_names.append(
                    f"Available Charging Station<br>"
                    f"{station.name}<br>"
                    f"ID: {station.station_id}<br>"
                    f"Power: {station.charging_power_kw} kW<br>"
                    f"Avg Availability: {avg_avail:.1f} chargers<br>"
                    f"Avg Speed: {avg_speed:.1f} mph"
                )

        if other_station_lats:
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=other_station_lons,
                lat=other_station_lats,
                marker={'size': 8, 'color': 'lightblue', 'symbol': 'circle',
                        'opacity': 0.6},
                text=other_station_names,
                name=f'Available Stations ({len(other_station_lats)})',
                showlegend=True
            ))

        all_lats = []
        all_lons = []
        for path in solutions[:5]:
            if path:
                for node in path:
                    node_data = graph.nodes[node]
                    all_lats.append(node_data['lat'])
                    all_lons.append(node_data['lon'])

        center_lat = sum(all_lats) / len(all_lats) if all_lats else 41.8781
        center_lon = sum(all_lons) / len(all_lons) if all_lons else -87.6298

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11
            ),
            height=800,
            title={
                'text': "EV Route Comparison with Charging Stations",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1
            )
        )

        return fig

    @staticmethod
    def plot_objectives_detail(objectives: List[PathObjectives],
                               title: str = "Route Objectives Comparison"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].bar(range(len(objectives)),
                       [obj.distance_km for obj in objectives],
                       color='steelblue')
        axes[0, 0].set_xlabel('Solution Index')
        axes[0, 0].set_ylabel('Distance (km)')
        axes[0, 0].set_title('Total Distance')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].bar(range(len(objectives)),
                       [obj.travel_time_min for obj in objectives],
                       color='coral')
        axes[0, 1].set_xlabel('Solution Index')
        axes[0, 1].set_ylabel('Time (min)')
        axes[0, 1].set_title('Travel Time')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].bar(range(len(objectives)),
                       [obj.charging_time_min for obj in objectives],
                       color='lightgreen')
        axes[1, 0].set_xlabel('Solution Index')
        axes[1, 0].set_ylabel('Time (min)')
        axes[1, 0].set_title('Charging Time')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(range(len(objectives)),
                       [obj.num_charging_stops for obj in objectives],
                       color='mediumpurple')
        axes[1, 1].set_xlabel('Solution Index')
        axes[1, 1].set_ylabel('Number of Stops')
        axes[1, 1].set_title('Charging Stops')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

    @staticmethod
    def create_route_summary(graph: nx.DiGraph,
                             path: List[str],
                             objective: PathObjectives,
                             route_planner,
                             route_name: str = "Route"):
        summary = []
        summary.append(f"{route_name.upper()} SUMMARY")

        # Overall statistics
        summary.append(f"\nOVERALL:")
        summary.append(f" Total Distance: {objective.distance_km:.2f} km")
        summary.append(f" Travel Time: {objective.travel_time_min:.1f} minutes ({objective.travel_time_min/60:.2f} hours)")
        summary.append(f" Charging Time: {objective.charging_time_min:.1f} minutes ({objective.charging_time_min/60:.2f} hours)")
        summary.append(f" Total Time: {objective.travel_time_min + objective.charging_time_min:.1f} minutes")
        summary.append(f" Number of Charging Stops: {objective.num_charging_stops}")
        summary.append(f" Final Battery SOC: {objective.final_soc * 100:.1f}%")

        charging_stops = route_planner.get_charging_stops_in_path(path)

        if charging_stops:
            summary.append(f"\nCHARGING STOPS ({len(charging_stops)}):")
            for i, node in enumerate(charging_stops, 1):
                stations = route_planner.get_stations_at_node(node)
                if stations:
                    station = stations[0]
                    avg_avail = RouteVisualizer._get_avg_availability(route_planner, station.station_id)
                    summary.append(f"\n  Stop {i}:")
                    summary.append(f" Location: {node}")
                    summary.append(f" Station: {station.name}")
                    summary.append(f" Station ID: {station.station_id}")
                    summary.append(f" Charging Power: {station.charging_power_kw} kW")
                    summary.append(f" Avg Availability: {avg_avail:.1f} chargers")
        else:
            summary.append(f"\nCHARGING STOPS: None (sufficient battery)")

        summary.append(f"\nPATH:")
        summary.append(f"  Number of nodes: {len(path)}")
        summary.append(f"  Start: {path[0]}")
        summary.append(f"  End: {path[-1]}")

        return "\n".join(summary)

    @staticmethod
    def plot_network(graph, stations: Dict = None, title: str = "Road Network"):
        fig = go.Figure()

        edge_lats = []
        edge_lons = []

        for u, v in graph.edges():
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]

            if u_data.get('lat') and v_data.get('lat'):
                edge_lats.extend([u_data['lat'], v_data['lat'], None])
                edge_lons.extend([u_data['lon'], v_data['lon'], None])

        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=edge_lons,
            lat=edge_lats,
            line={'width': 1, 'color': 'gray'},
            name='Roads',
            hoverinfo='skip'
        ))

        node_lats = []
        node_lons = []
        node_texts = []

        for node in graph.nodes():
            data = graph.nodes[node]
            if data.get('lat'):
                node_lats.append(data['lat'])
                node_lons.append(data['lon'])
                node_texts.append(f"Node: {node}")

        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=node_lons,
            lat=node_lats,
            marker={'size': 4, 'color': 'blue'},
            text=node_texts,
            name=f'Nodes ({len(node_lats)})'
        ))

        if stations:
            station_lats = []
            station_lons = []
            station_texts = []

            for station in stations.values():
                if station.src_node and station.src_node in graph:
                    data = graph.nodes[station.src_node]
                    if data.get('lat'):
                        station_lats.append(data['lat'])
                        station_lons.append(data['lon'])
                        station_texts.append(f"{station.name}<br>ID: {station.station_id}")

            if station_lats:
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=station_lons,
                    lat=station_lats,
                    marker={'size': 8, 'color': 'green', 'symbol': 'circle'},
                    text=station_texts,
                    name=f'Stations ({len(station_lats)})'
                ))

        center_lat = np.mean(node_lats) if node_lats else 41.8781
        center_lon = np.mean(node_lons) if node_lons else -87.6298

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10
            ),
            height=800,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_3d_pareto_front(solutions: List[List],
                             objectives: List[PathObjectives],
                             title: str = "NSGA-II Optimisation",
                             start_node=None, end_node=None):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if start_node and end_node:
            header = f"Multi-Objective EV Routing: Node {start_node} -> Node {end_node}\n{title}"
        else:
            header = f"Multi-Objective EV Routing\n{title}"
        fig.suptitle(header, fontsize=13)

        x = np.array([o.distance_km for o in objectives])
        y = np.array([o.travel_time_min for o in objectives])
        z = np.array([o.charging_time_min for o in objectives])
        c = np.array([o.num_charging_stops for o in objectives], dtype=float)

        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Travel Time (min)')
        ax.set_zlabel('Charging Time (min)')

        scatter = ax.scatter(x, y, z, c=c, cmap='hot', s=60, alpha=0.85)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Number of Charging Stops')
        cbar.set_ticks(np.arange(int(c.min()), int(c.max()) + 1))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_parallel_coordinates(objectives: List[PathObjectives],
                                  title: str = "EV Route Objectives - Parallel Coordinates"):
        df = pd.DataFrame({
            'Distance (km)': [o.distance_km for o in objectives],
            'Travel Time (min)': [o.travel_time_min for o in objectives],
            'Charging Time (min)': [o.charging_time_min for o in objectives],
            'Charging Stops': [o.num_charging_stops for o in objectives],
            'Final SOC (%)': [o.final_soc * 100 for o in objectives],
        })

        stops = df['Charging Stops']
        stop_ticks = list(range(int(stops.min()), int(stops.max()) + 1))

        fig = px.parallel_coordinates(
            df,
            color='Charging Stops',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={col: col for col in df.columns},
            title=title
        )
        for i, dim in enumerate(fig.data[0].dimensions):
            if dim.label == 'Charging Stops':
                fig.data[0].dimensions[i].tickvals = stop_ticks
                fig.data[0].dimensions[i].ticktext = [str(t) for t in stop_ticks]
                break
        fig.update_coloraxes(colorbar_tickvals=stop_ticks,
                             colorbar_ticktext=[str(t) for t in stop_ticks])
        return fig

    @staticmethod
    def plot_route_networkx(graph: nx.DiGraph, routes: List[List],
                            start_node, end_node,
                            labels: Optional[List[str]] = None,
                            route_planner=None,
                            output_path: Optional[str] = None):
        colours = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4']
        route_widths = [3.5, 3.5, 3.5, 3.5, 3.5]

        G2 = nx.DiGraph()
        for route in routes:
            if not route:
                continue
            route_edges = [(route[n], route[n + 1]) for n in range(len(route) - 1)]
            G2.add_nodes_from(route)
            G2.add_edges_from(route_edges)

        pos = {}
        for node in G2.nodes:
            if node in graph.nodes:
                lon = graph.nodes[node].get('lon')
                lat = graph.nodes[node].get('lat')
                if lon is not None and lat is not None:
                    pos[node] = [lon, lat]

        charging_nodes = set()
        if route_planner:
            for route in routes:
                if route:
                    charging_nodes.update(route_planner.get_charging_stops_in_path(route))

        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        ax.set_facecolor('white')

        for i, route in enumerate(routes):
            if not route:
                continue
            route_edges = [(route[n], route[n + 1]) for n in range(len(route) - 1)
                           if route[n] in pos and route[n + 1] in pos]
            nx.draw_networkx_edges(G2, pos=pos, ax=ax, edgelist=route_edges,
                                   edge_color=colours[i % len(colours)],
                                   arrows=True, width=route_widths[i % len(route_widths)],
                                   alpha=0.85, arrowsize=15,
                                   connectionstyle='arc3,rad=0.08')

        for i, route in enumerate(routes):
            if not route:
                continue
            valid_nodes = [n for n in route if n in pos and n not in charging_nodes
                           and n != start_node and n != end_node]
            nx.draw_networkx_nodes(G2, pos=pos, ax=ax, nodelist=valid_nodes,
                                   node_color=colours[i % len(colours)],
                                   node_size=60, alpha=0.9, linewidths=0.5,
                                   edgecolors='white')

        if charging_nodes:
            valid_charging = [n for n in charging_nodes if n in pos]
            xs = [pos[n][0] for n in valid_charging]
            ys = [pos[n][1] for n in valid_charging]
            ax.scatter(xs, ys, marker='*', s=500, c='gold', edgecolors='darkorange',
                       linewidths=1.5, zorder=6, label=f'Charging Stops ({len(valid_charging)})')

        for special, colour, size, marker in [
            (start_node, '#00cc44', 250, 'o'),
            (end_node,   '#cc0000', 250, 's'),
        ]:
            if special in pos:
                ax.scatter([pos[special][0]], [pos[special][1]],
                           s=size, c=colour, marker=marker,
                           edgecolors='black', linewidths=1.5, zorder=7)

        legend_items = []
        for i, route in enumerate(routes):
            if not route:
                continue
            label = labels[i] if labels else f"Route {i + 1}"
            legend_items.append(
                plt.Line2D([0], [0], color=colours[i % len(colours)],
                           linewidth=2.5, label=label)
            )
        legend_items += [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00cc44',
                       markeredgecolor='black', markersize=12, label=f'Start ({start_node})'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#cc0000',
                       markeredgecolor='black', markersize=12, label=f'End ({end_node})'),
        ]
        if charging_nodes:
            legend_items.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                           markeredgecolor='darkorange', markersize=14,
                           label=f'Charging Stops ({len([n for n in charging_nodes if n in pos])})'),
            )

        ax.legend(handles=legend_items, loc='upper left', fontsize=10,
                  framealpha=0.95, edgecolor='grey', fancybox=True)
        ax.set_title(f'Route Comparison: {start_node} -> {end_node}', fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3, color='grey')
        ax.tick_params(labelsize=9)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        return fig