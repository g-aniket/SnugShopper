from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np

from utils import (
    ellipse_circumference,
    get_chest,
    get_metric_per_pixel_customer,
    get_belly,
)


def get_body_measurements(height, front, side, debug=False):
    detector_front = PoseDetector(
        staticMode=True,
        modelComplexity=2,
        smoothLandmarks=False,
        enableSegmentation=True,
        smoothSegmentation=False,
        detectionCon=0.5,
        trackCon=0.5,
    )
    detector_side = PoseDetector(
        staticMode=True,
        modelComplexity=2,
        smoothLandmarks=False,
        enableSegmentation=True,
        smoothSegmentation=False,
        detectionCon=0.5,
        trackCon=0.5,
    )

    if front.shape[0] > 1024:
        front = cv2.resize(
            front, (int(front.shape[1] * 0.25), int(front.shape[0] * 0.25))
        )
    if side.shape[0] > 1024:
        side = cv2.resize(side, (int(side.shape[1] * 0.25), int(side.shape[0] * 0.25)))

    front = detector_front.findPose(front)
    front_lmlist, _, front_results = detector_front.findPosition(front)
    side = detector_side.findPose(side)
    side_lmlist, _, side_results = detector_side.findPosition(side)

    if front_lmlist is None:
        return {"error": "Cannot detect human body in the front image"}
    if side_lmlist is None:
        return {"error": "Cannot detect human body in the side image"}

    mpp_front = get_metric_per_pixel_customer(front_lmlist, height)
    mpp_side = get_metric_per_pixel_customer(side_lmlist, height)

    chest_front = (
        front_lmlist[24][0],
        int((front_lmlist[24][1] - front_lmlist[12][1]) * 0.333 + front_lmlist[12][1]),
    )
    belly_front = (
        front_lmlist[24][0],
        int((front_lmlist[24][1] - front_lmlist[12][1]) * 0.666 + front_lmlist[12][1]),
    )
    chest_side = (
        side_lmlist[24][0],
        int((side_lmlist[24][1] - side_lmlist[12][1]) * 0.333 + side_lmlist[12][1]),
    )
    belly_side = (
        side_lmlist[24][0],
        int((side_lmlist[24][1] - side_lmlist[12][1]) * 0.666 + side_lmlist[12][1]),
    )

    belly_major = get_belly(belly_front, front_results.segmentation_mask, mpp_front)
    belly_minor = get_belly(belly_side, side_results.segmentation_mask, mpp_side)
    belly_circumference = ellipse_circumference(belly_major / 2, belly_minor / 2)
    belly = ((belly_major) + (belly_circumference / 2)) * 1.03

    chest_major = get_chest(chest_front, front_results.segmentation_mask, mpp_front)
    chest_minor = get_chest(chest_side, side_results.segmentation_mask, mpp_side)
    chest_circumference = ellipse_circumference(chest_major / 2, chest_minor / 2)
    chest = (chest_major) + (chest_circumference / 2)

    shoulder = abs(front_lmlist[11][0] - front_lmlist[12][0]) * 1.06 * mpp_front

    p1 = front_lmlist[12][:2]
    p2 = front_lmlist[14][:2]
    p3 = front_lmlist[16][:2]
    p1_p2 = np.linalg.norm(np.array(p1) - np.array(p2)) * mpp_front
    p2_p3 = np.linalg.norm(np.array(p2) - np.array(p3)) * mpp_front
    arm_len = p1_p2 + p2_p3
    length = abs(front_lmlist[24][1] - front_lmlist[12][1]) * mpp_front

    if debug:
        front = cv2.circle(front, chest_front, 5, (0, 0, 0), -1)
        front = cv2.circle(front, belly_front, 5, (0, 0, 0), -1)
        side = cv2.circle(side, chest_side, 5, (0, 0, 0), -1)
        side = cv2.circle(side, belly_side, 5, (0, 0, 0), -1)
        front = cv2.circle(front, p1, 5, (0, 0, 0), -1)
        front = cv2.circle(front, p2, 5, (0, 0, 0), -1)
        front = cv2.circle(front, p3, 5, (0, 0, 0), -1)
        cv2.imshow("front", front)
        cv2.imshow("side", side)
        cv2.imshow("front_mask", front_results.segmentation_mask)
        cv2.imshow("side_mask", side_results.segmentation_mask)
        cv2.waitKey(0)

    return {
        "shoulder": float(round(shoulder, 2)),
        "chest": float(round(chest, 2)),
        "belly": float(round(belly, 2)),
        "sleeve": float(round(arm_len, 2)),
        "length": float(round(length, 2)),
    }
