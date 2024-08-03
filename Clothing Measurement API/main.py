from typing import List
import cv2
import numpy as np
from utils import draw_points_on_image, get_measurements


CONTRAST = 1  # Experiment with values between 1 (no change) and 10 (high contrast)
BRIGHTNESS = 1  # Adjust brightness as needed (positive values increase brightness)


def resize_with_aspect_ratio(image, width):
    (h, w) = image.shape[:2]
    r = width / float(w)
    new_height = int(h * r)
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def adjust_brightness_contrast(image, alpha, beta):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image


def blur_image_n_times(image, n, blur_radius=5):
    for _ in range(n):
        image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    return image


def detect_edges(image):
    kernel = np.ones((2, 2))
    image = cv2.Canny(image, 100, 100)
    image = cv2.dilate(image, kernel, iterations=3)
    return cv2.erode(image, kernel, iterations=3)


def get_all_contours(image):
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv2.contourArea)
    return contours


def get_shirt_contour(contours: List):
    for c in contours:
        if cv2.contourArea(c) > 40000:
            return c
    return None


def get_note_contour(contours: List):
    for c in contours:
        if cv2.contourArea(c) > 3000 and cv2.contourArea(c) < 20000:
            return c
    return None


def get_measurements_from_image(img, debug=False):
    img = resize_with_aspect_ratio(img, 800)
    adjusted_image = adjust_brightness_contrast(img.copy(), CONTRAST, BRIGHTNESS)
    r, g, b = cv2.split(adjusted_image)
    blank_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    blurredr = blur_image_n_times(r, 3)
    blurredg = blur_image_n_times(g, 3)
    blurredb = blur_image_n_times(b, 3)

    erodedr = detect_edges(blurredr)
    erodedg = detect_edges(blurredg)
    erodedb = detect_edges(blurredb)
    eroded = cv2.bitwise_or(erodedr, erodedg)
    eroded = cv2.bitwise_or(eroded, erodedb)

    if debug:
        cv2.imshow("eroded", eroded)

    contours = get_all_contours(eroded)
    note_contour = get_note_contour(contours)
    shirt_contour = get_shirt_contour(contours)

    if debug:
        blank_image = cv2.drawContours(
            blank_image, [shirt_contour], 0, (255, 255, 255), 1
        )
        blank_image = cv2.drawContours(
            blank_image, [note_contour], 0, (255, 255, 255), 1
        )
        cv2.imshow("shirt contour", blank_image)

    if note_contour is None:
        return {"error": True, "message": "Could not detect note. Please retry"}
    if shirt_contour is None:
        return {"error": True, "message": "Could not detect shirt. Please retry"}

    note_rect = cv2.boundingRect(note_contour)
    note_width = note_rect[2]
    real_width = 14.2  # centimeters
    metric_per_pixel = real_width / note_width
    print("metric_per_pixel", metric_per_pixel)

    blank_image = cv2.drawContours(blank_image, [shirt_contour], 0, (255, 255, 255), 1)
    blank_image = cv2.drawContours(blank_image, [note_contour], 0, (255, 255, 255), 1)

    gfttr = cv2.goodFeaturesToTrack(erodedr, 1000, 0.05, 5)
    gfttg = cv2.goodFeaturesToTrack(erodedg, 1000, 0.05, 5)
    gfttb = cv2.goodFeaturesToTrack(erodedb, 1000, 0.05, 5)

    gftt = np.concatenate((gfttr, gfttg), axis=0)
    gftt = np.concatenate((gftt, gfttb), axis=0)
    good_features_to_track = np.int_(gftt)

    points = good_features_to_track.reshape(-1, 2)

    for p in points:
        if p[1] <= note_rect[1] + note_rect[3]:
            points = np.delete(points, np.where(points == p), axis=0)

    if debug:
        boundingbox = cv2.boundingRect(points)
        blank_image = cv2.rectangle(
            blank_image,
            (boundingbox[0], boundingbox[1]),
            (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
            (255, 255, 255),
            1,
        )
        img = draw_points_on_image(img, points)
        cv2.imshow("bounding box", img)
        cv2.waitKey(0)

    measurements = get_measurements(points, metric_per_pixel)
    return measurements
