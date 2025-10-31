import requests
from fiona.features import distance
from lxml import html
import csv

url = 'http://iot.ee.surrey.ac.uk:8080/datasets/traffic/traffic_feb_june/index.html'
response = requests.get(url)

if response.status_code == 200:
    tree = html.fromstring(response.content)
    rows = tree.xpath('//table//tr')[1:]
    data = []

    for row in rows:
        cols = row.xpath('.//td')
        if len(cols) >= 7:

            name = cols[0].text_content().strip()
            duration_from = cols[1].text_content().strip()
            duration_to = cols[2].text_content().strip()
            point_1 = cols[5].text_content().strip()
            point_2 = cols[6].text_content().strip()

            city_1 = point_1[point_1.find("City:") + len("City:"):point_1.find("Street:")].strip()
            street_1 = point_1[point_1.find("Street:") + len("Street:"):point_1.find("Postal Code:")].strip()
            postal_code_1 = point_1[point_1.find("Postal Code:") + len("Postal Code:"):point_1.find("Coordinates (lat, long):")].strip()
            coordinates_1 = point_1[point_1.find("Coordinates (lat, long):") + len("Coordinates (lat, long):"):].strip()

            city_2 = point_2[point_2.find("City:") + len("City:"):point_2.find("Street:")].strip()
            street_2 = point_2[point_2.find("Street:") + len("Street:"):point_2.find("Postal Code:")].strip()
            postal_code_2 = point_2[point_2.find("Postal Code:") + len("Postal Code:"):point_2.find("Coordinates (lat, long):")].strip()
            coordinates_2 = point_2[point_2.find("Coordinates (lat, long):") + len("Coordinates (lat, long):"):].strip()


            point_data = cols[7].text_content().strip()
            distance = point_data[point_data.find("Distance between two points in meters:") + len("Distance between two points in meters:"):point_data.find("Duration of measurements in seconds:")].strip()
            duration = point_data[point_data.find("Duration of measurements in seconds:") + len("Duration of measurements in seconds:"):point_data.find("NDT in KMH:")].strip()
            ndt = point_data[point_data.find("NDT in KMH:") + len("NDT in KMH:"):point_data.find("EXT ID:")].strip()
            road_type = point_data[point_data.find("Road type:") + len("Road type:"):point_data.find("Report ID and Report Name:")].strip()

            data.append((name,duration_from,duration_to,city_1,street_1,postal_code_1,coordinates_1,city_2,street_2,postal_code_2,coordinates_2,distance,duration,ndt,road_type))

    with open('observation_metadata.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Duration From", "Duration To", "Observation 1 City", "Observation 1 Street", "Observation 1 Postal Code", "Observation 1 Coordinates", "Observation 2 City", "Observation 2 Street", "Observation 2 Postal Code", "Observation 2 Coordinates", "Distance(meters)", "Duration(seconds)", "NDT", "Road Type"])
        for item in data:
            writer.writerow(item)

    print("Data has been saved to 'observation_metadata.csv'.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
