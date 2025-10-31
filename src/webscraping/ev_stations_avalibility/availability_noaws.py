import json
import http.client
import datetime
import os


def fetch_and_save_chargehub_data():
    conn = http.client.HTTPSConnection("apiv2.chargehub.com")

    # Parametri della richiesta (puoi modificarli se vuoi cambiare area geografica o filtri)
    endpoint = (
        "/api/locationsmap/v2"
        "?latmin=49.673407876197295"
        "&latmax=50.15094937313377"
        "&lonmin=-98.69580115772173"
        "&lonmax=-94.74072303272173"
        "&limit=250"
        "&key=olitest"
        "&only_passport=0"
        "&remove_networks="
        "&remove_levels=1"
        "&remove_connectors="
        "&free=0"
        "&above_power=0"
        "&show_pending=1"
        "&only_247=0"
    )

    conn.request("GET", endpoint)
    response = conn.getresponse()
    data = response.read().decode("utf-8")
    conn.close()

    # Decodifica del contenuto JSON
    json_data = json.loads(data)

    # Timestamp per il nome file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{timestamp}.json"

    # Salvataggio su disco locale
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Dati salvati in {filename}")


# Esecuzione diretta se lo script è lanciato manualmente
if __name__ == "__main__":
    fetch_and_save_chargehub_data()
