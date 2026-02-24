
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import List, Dict

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
        total_times = [obj.travel_time_min + obj.charging_time_min for obj in objectives]

        ax = fig.add_subplot(111)

        ax.scatter(distances, total_times, c='b', s=100, alpha=0.7, edgecolors='k')

        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Total Time (min)', fontsize=12)
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

        for node in used_charging_nodes:
            if node in graph.nodes:
                node_data = graph.nodes[node]
                used_station_lats.append(node_data['lat'])
                used_station_lons.append(node_data['lon'])

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
                marker={'size': 22, 'color': '#FF4500', 'symbol': 'star',  # OrangeRed
                   },
                text=used_station_names,
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

        center_lat = sum(all_lats) / len(all_lats) if all_lats else 40.7128
        center_lon = sum(all_lons) / len(all_lons) if all_lons else -74.0060

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

        center_lat = np.mean(node_lats) if node_lats else 40.7128
        center_lon = np.mean(node_lons) if node_lons else -74.0060

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