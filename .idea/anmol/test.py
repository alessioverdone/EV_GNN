import pandas as pd

folder_path = "/Users/anmolpreetsingh/Desktop/trafficData.csv"
traffic_data = pd.read_csv(folder_path)

print(traffic_data.columns)