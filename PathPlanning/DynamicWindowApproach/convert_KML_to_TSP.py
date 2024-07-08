import xml.etree.ElementTree as ET

def parse_kml(kml_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    coordinates = []

    for placemark in root.findall('.//kml:Placemark', namespace):
        coords = placemark.find('.//kml:coordinates', namespace).text.strip()
        for coord in coords.split():
            lon, lat, _ = map(float, coord.split(','))
            coordinates.append((lat, lon))

    return coordinates

def convert_to_tsp(coordinates, tsp_file):
    with open(tsp_file, 'w') as f:
        f.write(f"NAME: example\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {len(coordinates)}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")

        for i, (lat, lon) in enumerate(coordinates, start=1):
            f.write(f"{i} {lat} {lon}\n")

        f.write(f"EOF\n")

# Main execution
kml_file = '/home/dell/ALNS/Fences.kml'
tsp_file = '/home/dell/ALNS/Fences.tsp'

coordinates = parse_kml(kml_file)
convert_to_tsp(coordinates, tsp_file)

print(f"Converted {len(coordinates)} points from {kml_file} to {tsp_file}")

