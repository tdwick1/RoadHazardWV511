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
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class CameraEntry:
    title: str
    cam_id: str
    raw: str  # raw myCams[n] entry for debugging / future parsing


class WV511CameraClient:
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
        return html, self._soup(html)
    
    def get_all_routes(self, timeout: int = 30) -> List[str]:
        html, soup = self.get_listing_page(timeout=timeout)
        routes: List[str] = []
        select = soup.find("select", {"name": "ctl00$ContentPlaceHolder1$ddlRoutes"})
        if not select:
            return routes
        for option in select.find_all("option"):
            value = option.attrs.get('value')
            if not value:
                continue
            routes.append(value.strip())
        return routes, html, soup

    def postback_select_route(
        self,
        route_value: str,
        ddl_name: str = "ctl00$ContentPlaceHolder1$ddlRoutes",
        timeout: int = 30,
        listing_page: Optional[Tuple[str, BeautifulSoup]] = None,
    ) -> Tuple[str, BeautifulSoup]:
        """
        Mimic changing the route dropdown on CameraListing.aspx.
        Returns the HTML and soup after the postback.
        """
        if listing_page is None:
            html, soup = self.get_listing_page(timeout=timeout)
        else:
            html, soup = listing_page
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
        listing_page: Optional[Tuple[str, BeautifulSoup]] = None,
    ) -> List[CameraEntry]:
        """
        Returns camera entries (title + cam_id) for a given route dropdown value,
        e.g. "US-60".
        """
        html, soup = self.postback_select_route(
            route_value=route_value, ddl_name=ddl_name, timeout=timeout, listing_page=listing_page
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

    client = WV511CameraClient()
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
