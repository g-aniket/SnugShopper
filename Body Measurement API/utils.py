import cv2
from cv2.typing import MatLike, Point
import numpy as np


def draw_points_on_image(
    image: MatLike,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
):
    """Draws points on image"""
    for point in points:
        x, y = point
        image = cv2.circle(image, (x, y), radius, color, -1)
    return image


def get_bottom_left_right(points):
    """Returns bottom left and right points"""
    bottom_left = max(points, key=lambda p: p[1] - p[0])
    bottom_right = max(points, key=lambda p: p[0] + p[1])
    return bottom_left, bottom_right


def get_left_shoulder(points):
    """Returns left shoulder point"""
    left_shoulder = min(points, key=lambda p: p[0] - p[1])
    return left_shoulder


def get_right_shoulder(points):
    """Returns right shoulder point"""
    right_shoulder = min(points, key=lambda p: p[0] + p[1])
    return right_shoulder


def calculate_width(points, metric_per_pixel):
    """Returns width"""
    bottom_left, bottom_right = points["bottom_left"], points["bottom_right"]
    dist = (
        np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
        * metric_per_pixel
    )
    return float(dist)


def calculate_shoulder_length(points, metric_per_pixel):
    """Returns shoulder length"""
    left_shoulder = get_left_shoulder(points)
    right_shoulder = get_right_shoulder(points)
    dist = (
        np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        * metric_per_pixel
    )
    return float(dist)


def get_collar_points(points: list[tuple[int, int]]):
    """Returns top 2 points as collar points"""
    p = sorted(points, key=lambda p: p[1])
    collars = [p[0], p[1]]
    collars = sorted(collars, key=lambda p: p[0])
    return collars


def get_sleeve_points(points: list[tuple[int, int]]):
    """Returns top 2 points as collar points"""
    l = sorted(points, key=lambda p: p[0])
    r = sorted(points, key=lambda p: p[0], reverse=True)
    left_sleeve = l[0]
    right_sleeve = r[0]
    return left_sleeve, right_sleeve


def calculate_collar(points, metric_per_pixel):
    """Returns length"""
    left_collar, right_collar = points["left_collar"], points["right_collar"]
    dist = (
        np.linalg.norm(np.array(left_collar) - np.array(right_collar))
        * metric_per_pixel
    )
    return float(dist)


def calculate_sleeve(points, metric_per_pixel):
    """Returns length"""
    left_sleeve, right_sleeve = points["left_most"].copy(), points["right_most"].copy()
    total_dist = np.linalg.norm(np.array(left_sleeve) - np.array(right_sleeve))
    sleeve = total_dist - (points["bottom_right"][0] - points["bottom_left"][0])
    dist = (sleeve / 2) * 1.1 * metric_per_pixel
    return float(dist)


def calculate_length(points, metric_per_pixel):
    """Returns length"""
    bottom_left, collar_left = points["bottom_left"].copy(), points["left_collar"]
    bottom_left[0] = collar_left[0]
    dist = (
        np.linalg.norm(np.array(collar_left) - np.array(bottom_left)) * metric_per_pixel
    )
    return float(dist)


def get_measurements(points, metric_per_pixel):
    """Returns measurements from points"""
    measurements = {}
    poi = {}
    poi["bottom_left"], poi["bottom_right"] = get_bottom_left_right(points)
    poi["left_collar"], poi["right_collar"] = get_collar_points(points)
    poi["left_most"], poi["right_most"] = get_sleeve_points(points)

    width = calculate_width(poi, metric_per_pixel)
    measurements["length"] = calculate_length(poi, metric_per_pixel)
    measurements["shoulder"] = width * 0.85
    measurements["sleeve"] = calculate_sleeve(poi, metric_per_pixel)
    measurements["chest"] = width * 2
    measurements["belly"] = width * 2
    return measurements


def get_metric_per_pixel_customer(lmlist, real_height):
    """Returns metric per pixel"""
    height = lmlist[30][1] - lmlist[2][1]
    return real_height / height


def get_shoulder_length(lmlist, metric_per_pixel):
    """Returns shoulder length"""
    left_x = lmlist[12][0]
    right_x = lmlist[11][0]
    dist = abs(right_x - left_x) * 1.075 * metric_per_pixel
    return float(dist)


def get_chest(chest_l, segmentation_mask, metric_per_pixel):
    """Returns chest length"""
    threshold = 0.5
    row = segmentation_mask[chest_l[1]]
    left = np.where(row[: chest_l[0]] < threshold)[0][-1]
    right = np.where(row[chest_l[0] :] < threshold)[0][0] + chest_l[0]
    dist = abs(right - left) * metric_per_pixel
    return float(dist)


def get_belly(belly_l, segmentation_mask, metric_per_pixel):
    """Returns shoulder length"""
    threshold = 0.5
    row = segmentation_mask[belly_l[1]]
    left = np.where(row[: belly_l[0]] < threshold)[0][-1]
    right = np.where(row[belly_l[0] :] < threshold)[0][0] + belly_l[0]
    dist = abs(right - left) * metric_per_pixel
    return float(dist)


def ellipse_circumference(major, minor):
    """Returns circumference"""
    return np.pi * (1.5 * (major + minor) - np.sqrt(major * minor))
