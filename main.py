import os
from wv511 import WV511Client

client = WV511Client()
cams, debug = client.get_cameras_between_addresses(
    "127 Ranch Lake Blvd, Scott Depot, WV 25560",
    "119 Riverview Dr, Poca, WV 25159",
    max_miles_from_route=5,
    return_debug=True,
)
print(cams,debug)
for d in debug[:20]:
    print(f"{d['miles_from_route']:.2f} mi | {d['cam_id']} | {d['title']} | {d['query']}")


# routes = client.get_all_routes() # returns list of route strings (e.g. "US-60")
# # route = routes[1]
# print(routes)
# exit()
# for route in routes:
#     print(route)
#     cams = client.get_cameras_for_route(route) # returns list of camera entries for the given route, contains cam_id and title

#     for cam in cams:
#         image = client.fetch_camera_image(cam.cam_id)
#         if not image:
#             print(f"Failed to fetch image for camera {cam.cam_id}.")
#             continue

#         folder_name = f"snapshots/{route.replace('-', '_').replace(' ', '_').replace('.','')}"
#         if not os.path.exists(folder_name):
#             os.makedirs(folder_name)
#         file_name = f"{folder_name}/camera_snapshot_{cam.cam_id}.jpg"

#         with open(file_name, "wb") as f:
#             f.write(image)

#         print(f"Saved image to {file_name}")