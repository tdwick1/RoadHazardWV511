import googlemaps
from datetime import datetime
import os

import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_directions(origin_address, destination_address):

    # Request directions via driving
    directions_result = gmaps_client.directions(origin_address,
                                         destination_address,
                                         mode="driving",
                                         departure_time=datetime.now())
    return directions_result

def geocode(address: str) -> Optional[Tuple[float, float]]:
    """Geocode an address string to (lat, lng) tuple using Google Maps API."""
    geocode_result = gmaps_client.geocode(address)
    if not geocode_result:
        return None
    location = geocode_result[0]['geometry']['location']
    return (location['lat'], location['lng'])

