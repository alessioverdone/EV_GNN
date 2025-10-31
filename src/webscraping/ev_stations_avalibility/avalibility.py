import json
import boto3
import http.client
import datetime

BUCKET = 'S3_BUCKET_NAME'


def lambda_handler(event, context):
    conn = http.client.HTTPSConnection("apiv2.chargehub.com")
    payload = ''
    headers = {}
    conn.request("GET",
                 "/api/locationsmap/v2?latmin=49.673407876197295&latmax=50.15094937313377&lonmin=-98.69580115772173&lonmax=-94.74072303272173&limit=250&key=olitest&only_passport=0&remove_networks=&remove_levels=1&remove_connectors=&free=0&above_power=0&show_pending=1&only_247=0%0A",
                 payload, headers)
    response = conn.getresponse()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = response.read().decode("utf-8")

    json_data = json.loads(data)
    client = boto3.client('s3')

    client.put_object(Body=json.dumps(json_data, indent=4), Bucket=BUCKET, Key=f"response_{timestamp}.json")
    print('Done uploading to S3')

    return {
        'statusCode': 200,
        'body': json.dumps('')
    }
