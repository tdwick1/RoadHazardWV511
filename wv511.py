"""
wv511_cams.py

A small client for WV511 CameraListing.aspx (ASP.NET WebForms) to:
- load routes
- postback to select a route
- extract camera IDs from the myCams JS array
- resolve a camera ID to its HLS (.m3u8) playlist URL via flowplayeri.aspx
"""

from __future__ import annotations

import re
import time
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set

import requests
from bs4 import BeautifulSoup

from gmaps import get_directions, geocode

import math




# -------------------------
# Geo helpers
# -------------------------

def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    index = lat = lng = 0
    length = len(encoded)

    while index < length:
        shift = result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))

    return coords


def haversine_miles(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    r = 3958.7613  # miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def nearest_polyline_vertex(poly: List[Tuple[float, float]], p: Tuple[float, float]) -> Tuple[float, int]:
    best_d = float("inf")
    best_i = -1
    for i, v in enumerate(poly):
        d = haversine_miles(v, p)
        if d < best_d:
            best_d = d
            best_i = i
    return best_d, best_i


# -------------------------
# Camera title -> geocode query
# -------------------------

def strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

_MM_RE = re.compile(r"\bMM\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_AT_RE = re.compile(r"@\s*(.+)$")  # everything after "@"

def camera_title_to_query(title: str) -> Optional[str]:
    if not title or title.strip().lower() == "junction":
        return None

    t = title
    t = re.sub(r"^[A-Z0-9\-]+:\s*", "", t)      # "I-64: "
    t = re.sub(r"\[[A-Z]{3}\]\s*", "", t)       # "[KAN]"
    t = t.replace("I/C", "interchange")

    # Identify road prefix at start (I-64 / US-35 / WV-34 etc) if present
    road_match = re.match(r"^((?:I|US|WV)-\d{1,3})\b", t.strip(), re.IGNORECASE)
    road = road_match.group(1).upper() if road_match else ""

    mm = _MM_RE.search(t)
    if mm and road:
        return f"{road} mile {mm.group(1)} West Virginia"

    m_at = _AT_RE.search(t)
    if m_at and road:
        after = m_at.group(1).strip()
        # If the "after" starts numeric, treat as mile marker
        mnum = re.match(r"^(\d+(?:\.\d+)?)\b", after)
        if mnum:
            return f"{road} mile {mnum.group(1)} West Virginia"
        # Else use the descriptive text (exit/interchange/place)
        return f"{road} {after} West Virginia"

    # fallback
    return f"{t} West Virginia"


@dataclass(frozen=True)
class CameraEntry:
    title: str
    cam_id: str
    raw: str  # raw myCams[n] entry for debugging / future parsing


class WV511Client:
    CAMERA_LISTING_URL = "https://www.wv511.org/CameraListing.aspx"
    FLOWPLAYER_URL = "https://www.wv511.org/flowplayeri.aspx"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self._apply_default_headers()

    def _apply_default_headers(self) -> None:
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/144.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,"
                    "application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "max-age=0",
                "Origin": "https://www.wv511.org",
                "Referer": self.CAMERA_LISTING_URL,
                "Upgrade-Insecure-Requests": "1",
            }
        )

    # -------------------------
    # Directions -> WV511 route extraction
    # -------------------------

    _ROUTE_TOKEN_RE = re.compile(
        r"""
        \b(
            I[-\s]?\d{1,3}          |   # I-64, I 64
            US[-\s]?\d{1,3}         |   # US-60, US 60
            U\.S\.\s?\d{1,3}        |   # U.S. 60
            WV[-\s]?\d{1,3}         |   # WV-34, WV 34
            W\.?V\.?\s?\d{1,3}          # W.V. 34
        )\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    @staticmethod
    def _strip_html(html: str) -> str:
        # Turns "Merge onto <b>I-64 E</b> ..." into plain text
        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

    @classmethod
    def _normalize_route_token(cls, token: str) -> Optional[str]:
        """
        Convert variations like:
          "I 64" / "i-64" -> "I-64"
          "U.S. 60" -> "US-60"
          "W.V. 34" -> "WV-34"
        into the WV511 dropdown-like form.
        """
        if not token:
            return None

        t = token.upper().strip()

        # Normalize punctuation/spacing for prefixes
        t = t.replace("U.S.", "US").replace("U. S.", "US")
        t = t.replace("W.V.", "WV").replace("W. V.", "WV")
        t = re.sub(r"\s+", " ", t)

        # Convert "I 64" -> "I-64", "US 60" -> "US-60", "WV 34" -> "WV-34"
        m = re.match(r"^(I|US|WV)[-\s]?(\d{1,3})$", t)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

        # Sometimes token might already be like "I-64"
        m = re.match(r"^(I|US|WV)-(\d{1,3})$", t)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

        return None

    @classmethod
    def _extract_route_values_from_directions(cls, directions: object) -> List[str]:
        """
        directions is whatever gmaps.get_directions() returns. Your example is:
          directions == [ { "legs": [...], "summary": "...", ... } ]

        We scan:
          - route["summary"]
          - every step["html_instructions"]
        and pull highway tokens (I-64, US-60, WV-xx), normalize, and dedupe.
        """
        if not directions:
            return []

        # If someone passed a single dict instead of list
        routes_iter: Iterable[dict]
        if isinstance(directions, dict):
            routes_iter = [directions]
        else:
            routes_iter = directions  # expected list

        found: List[str] = []
        seen: Set[str] = set()

        def add_token(tok: str) -> None:
            norm = cls._normalize_route_token(tok)
            if norm and norm not in seen:
                seen.add(norm)
                found.append(norm)

        for route in routes_iter:
            if not isinstance(route, dict):
                continue

            # 1) route-level summary (often contains the main interstate)
            summary = route.get("summary", "") or ""
            for m in cls._ROUTE_TOKEN_RE.finditer(summary):
                add_token(m.group(1))

            # 2) step instructions (usually richer)
            for leg in route.get("legs", []) or []:
                for step in leg.get("steps", []) or []:
                    instr = cls._strip_html(step.get("html_instructions", "") or "")
                    for m in cls._ROUTE_TOKEN_RE.finditer(instr):
                        add_token(m.group(1))

        return found


    # -------------------------
    # HTML / WebForms helpers
    # -------------------------

    @staticmethod
    def _soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    @staticmethod
    def _get_hidden_value(soup: BeautifulSoup, name: str) -> str:
        tag = soup.find("input", {"name": name})
        if tag and tag.has_attr("value"):
            return tag["value"]
        return ""

    def _get_form_tokens(self, soup: BeautifulSoup) -> dict:
        # These are the ones you actually needed in your script.
        # You can add __VIEWSTATEGENERATOR, etc., if WV511 starts requiring it.
        return {
            "__VIEWSTATE": self._get_hidden_value(soup, "__VIEWSTATE"),
            "__LASTFOCUS": self._get_hidden_value(soup, "__LASTFOCUS"),
            "__EVENTVALIDATION": self._get_hidden_value(soup, "__EVENTVALIDATION"),
        }

    @staticmethod
    def _get_selected_option_value(soup: BeautifulSoup, select_name: str) -> str:
        select = soup.find("select", {"name": select_name})
        if not select:
            return ""
        opt = select.find("option", selected=True)
        return opt["value"] if opt and opt.has_attr("value") else ""

    @staticmethod
    def _find_cam_data_script(soup: BeautifulSoup) -> str:
        # Looks for the script that defines: var myCams = new Array();
        for script in soup.find_all("script"):
            if script.string and "var myCams = new Array();" in script.string:
                return script.string
        return ""

    @staticmethod
    def _extract_js_array_entries(script: str, array_name: str) -> List[str]:
        # myCams[0] = "....";
        pattern = re.compile(rf'{re.escape(array_name)}\[\d+\]\s*=\s*"(.*)";')
        out: List[str] = []
        for line in script.splitlines():
            m = pattern.match(line.strip())
            if m:
                out.append(m.group(1))
        return out

    # -------------------------
    # Public API
    # -------------------------

    def get_listing_page(self, timeout: int = 30) -> Tuple[str, BeautifulSoup]:
        r = self.session.get(self.CAMERA_LISTING_URL, timeout=timeout)
        r.raise_for_status()
        html = r.text
        soup = self._soup(html)
        self._listing_page = (html, soup)
        return html, soup
    
    def _get_listing_page(self, timeout: int = 30) -> Tuple[str, BeautifulSoup]:
        if hasattr(self, '_listing_page'):
            return self._listing_page
        return self.get_listing_page(timeout=timeout)

    def get_all_routes(self, timeout: int = 30) -> List[str]:
        _, soup = self._get_listing_page(timeout=timeout)
        routes: List[str] = []
        select = soup.find("select", {"name": "ctl00$ContentPlaceHolder1$ddlRoutes"})
        if not select:
            return routes
        for option in select.find_all("option"):
            value = option.attrs.get('value')
            if not value:
                continue
            routes.append(value.strip())
        return routes

    def postback_select_route(
        self,
        route_value: str,
        ddl_name: str = "ctl00$ContentPlaceHolder1$ddlRoutes",
        timeout: int = 30,
    ) -> Tuple[str, BeautifulSoup]:
        """
        Mimic changing the route dropdown on CameraListing.aspx.
        Returns the HTML and soup after the postback.
        """
        _, soup = self._get_listing_page(timeout=timeout)
        tokens = self._get_form_tokens(soup)

        payload = {
            **tokens,
            "__EVENTTARGET": ddl_name,
            "__EVENTARGUMENT": "",
            ddl_name: route_value,
        }

        r2 = self.session.post(self.CAMERA_LISTING_URL, data=payload, timeout=timeout)
        r2.raise_for_status()
        html2 = r2.text
        return html2, self._soup(html2)

    def get_cameras_for_route(
        self,
        route_value: str,
        ddl_name: str = "ctl00$ContentPlaceHolder1$ddlRoutes",
        timeout: int = 30,
    ) -> List[CameraEntry]:
        """
        Returns camera entries (title + cam_id) for a given route dropdown value,
        e.g. "US-60".
        """
        html, soup = self.postback_select_route(
            route_value=route_value, ddl_name=ddl_name, timeout=timeout
        )

        selected = self._get_selected_option_value(soup, ddl_name)
        # Not fatal, but helpful if WV511 changes names/values.
        if selected and selected != route_value:
            # keep going anyway; sometimes selected matches, sometimes blank
            pass

        script = self._find_cam_data_script(soup)
        if not script:
            raise RuntimeError("Camera data script (myCams array) not found in response.")

        entries = self._extract_js_array_entries(script, "myCams")

        cams: List[CameraEntry] = []
        for raw in entries:
            # Example:
            # US-60: [KAN]US-60 @ Chelyan||/images/cam_unavailable.jpg|CAM090|1|0|0|
            parts = raw.split("|")
            if len(parts) < 4:
                continue
            title = parts[0]
            cam_id = parts[3]
            cams.append(CameraEntry(title=title, cam_id=cam_id, raw=raw))

        return cams

    def get_m3u8_url(self, cam_id: str, timeout: int = 30) -> Optional[str]:
        """
        Fetch flowplayeri.aspx and extract the HLS source (application/x-mpegurl).
        """
        nocache = int(time.time() * 1000)
        url = f"{self.FLOWPLAYER_URL}?CAMID={cam_id}&nocache={nocache}"

        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()

        soup = self._soup(r.text)
        source = soup.find("source", {"type": "application/x-mpegurl"})
        if source and source.has_attr("src"):
            return source["src"]
        return None
    
    def fetch_camera_image(self, cam_id: str, timeout: int = 30) -> Optional[bytes]:
        """
        Fetch m3u8 playlist and extract the image.
        """
        m3u8_url = self.get_m3u8_url(cam_id, timeout=timeout)
        if not m3u8_url:
            return None
        
        cap = cv2.VideoCapture(m3u8_url)
        success, frame = cap.read()

        cap.release()
        
        if success:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()

        return None
        
    def save_image_to_file(self, image_bytes: bytes, filename: str) -> None:
        """
        Save image bytes to a file.
        """
        with open(filename, "wb") as f:
            f.write(image_bytes)


    def fetch_m3u8_playlist_text(self, m3u8_url: str, timeout: int = 30) -> str:
        r = self.session.get(m3u8_url, timeout=timeout)
        r.raise_for_status()
        return r.text
    
    def get_cameras_between_addresses(
        self,
        origin: str,
        destination: str,
        *,
        max_miles_from_route: float = 1.0,
        timeout: int = 30,
        return_debug: bool = False,
    ):
        directions = get_directions(origin, destination)
        if not directions:
            return [] if not return_debug else ([], [])

        route0 = directions[0]
        enc = (route0.get("overview_polyline") or {}).get("points")
        if not enc:
            return [] if not return_debug else ([], [])

        poly = decode_polyline(enc)

        route_values = self._extract_route_values_from_directions(directions)
        if not route_values:
            return [] if not return_debug else ([], [])

        # Candidate cameras (dedupe)
        candidates: Dict[str, CameraEntry] = {}
        for rv in route_values:
            try:
                for cam in self.get_cameras_for_route(rv, timeout=timeout):
                    candidates.setdefault(cam.cam_id, cam)
            except Exception:
                continue

        geo_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        kept: List[Tuple[int, float, CameraEntry, str]] = []  # (poly_idx, dist, cam, query)

        for cam in candidates.values():
            q = camera_title_to_query(cam.title)
            if not q:
                continue

            if cam.cam_id not in geo_cache:
                geo_cache[cam.cam_id] = geocode(q)

            pt = geo_cache[cam.cam_id]
            if not pt:
                continue

            dmin, idx = nearest_polyline_vertex(poly, pt)
            if dmin <= max_miles_from_route:
                kept.append((idx, dmin, cam, q))

        kept.sort(key=lambda x: x[0])  # order along route

        cams_ordered = [cam for _, __, cam, ___ in kept]

        if not return_debug:
            return cams_ordered

        debug = [
            {"cam_id": cam.cam_id, "title": cam.title, "query": q, "miles_from_route": d}
            for _, d, cam, q in kept
        ]
        return cams_ordered, debug



# -------------------------
# Optional CLI usage:
#   python wv511_cams.py US-60
#   python wv511_cams.py US-60 --index 1
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WV511 camera scraper (route -> cams -> m3u8).")
    parser.add_argument("route", help='Route value in dropdown, e.g. "US-60"')
    parser.add_argument("--index", type=int, default=0, help="Camera index within route list")
    parser.add_argument("--print-playlist", action="store_true", help="Fetch and print playlist content")

    args = parser.parse_args()

    client = WV511Client()
    cams = client.get_cameras_for_route(args.route)

    if not cams:
        raise SystemExit(f"No cameras found for route={args.route}")

    idx = args.index
    if idx < 0 or idx >= len(cams):
        raise SystemExit(f"index out of range (0..{len(cams)-1})")

    cam = cams[idx]
    print(f"Camera[{idx}] {cam.title} -> {cam.cam_id}")

    m3u8 = client.get_m3u8_url(cam.cam_id)
    if not m3u8:
        raise SystemExit("Could not resolve m3u8 URL from flowplayeri.aspx")

    print(f"m3u8: {m3u8}")

    if args.print_playlist:
        text = client.fetch_m3u8_playlist_text(m3u8)
        print("\n--- playlist ---\n")
        print(text)
