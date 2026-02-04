from wv511 import WV511CameraClient

client = WV511CameraClient()

routes, html, soup = client.get_all_routes()
route = routes[1]
print(f"Selected route: {route}")
cams = client.get_cameras_for_route(route, listing_page=(html, soup))
print(f"Found {len(cams)} cameras on route {route}.")

for cam in cams:
    image = client.fetch_camera_image(cam.cam_id)
    if not image:
        print("Failed to fetch image.")
        exit()

    print('Fetched camera image!')

    file_name = f"snapshots/camera_snapshot_{cam.cam_id}.jpg"

    with open(file_name, "wb") as f:
        f.write(image)

    print(f"Saved image to {file_name}")