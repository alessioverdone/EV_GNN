import boto3
import json
import csv
from datetime import datetime
import re

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
BUCKET = ''


def extract_datetime_from_filename(filename):
    match = re.search(r"(\d{8}_\d{6})", filename)
    if match:
        dt_string = match.group(1)
        dt_obj = datetime.strptime(dt_string, "%Y%m%d_%H%M%S")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    return None

def filter_locations(json_data, locations_hash, file_name):
    places = json.loads(json_data)["locationsArray"]

    for location in places:
        loc_name = location["LocName"]
        available_ports = '0'
        total_ports = '0'

        for port in location["PortSummary"]:
            if port["StatusString"] == "Available":
                available_ports = port["CountString"]
            elif port["StatusString"] == "Total":
                total_ports = port["CountString"]
            elif port["StatusString"] == "Coming Soon":
                available_ports = "NA"
                total_ports = port["CountString"]

        if loc_name not in locations_hash:
            locations_hash[loc_name] = {}
        if file_name not in locations_hash[loc_name]:
            locations_hash[loc_name][file_name] = {}

        locations_hash[loc_name][file_name] = {
            "Ports": available_ports + "/" + total_ports
        }

def save_to_csv(location_hash, filename="filtered_locations.csv"):
    headers = ["Time"]
    for location in location_hash.keys():
        headers.append(location)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)

        data = {}
        for location, files in location_hash.items():
            for file_name, ports in files.items():
                if file_name not in data:
                    data[file_name] = {}

                data[file_name][location] = ports["Ports"]

        for file_name, places in data.items():
            time_list = [extract_datetime_from_filename(file_name)]

            for columns in headers[1:]:
                if columns in places:
                    time_list.append(places[columns])
                else:
                    time_list.append("")

            writer.writerow(time_list)

s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
objects = s3_client.list_objects_v2(Bucket=BUCKET)
locations = {}

i = 0
for obj in objects['Contents']:
    print(obj['Key'])
    response = s3_client.get_object(Bucket=BUCKET, Key=obj['Key'])
    content = response['Body'].read().decode('utf-8')
    filter_locations(content, locations, obj['Key'].replace("response_", "").replace(".json", ""))

save_to_csv(locations)