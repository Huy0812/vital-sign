from pathlib import Path
import warnings
import time

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from region_of_interest import RegionOfInterest, get_default_bgmask

resource_path = Path(__file__).parent.parent / "_resources"


def exponential_smooth(new_roi, old_roi, factor):
    if factor <= 0.0 or old_roi is None:
        return new_roi

    smooth_roi = np.multiply(new_roi, 1 - factor) + np.multiply(old_roi, factor)
    return tuple(smooth_roi.astype(int))

def get_boundingbox_from_landmarks(lms):
    xy = np.min(lms, axis=0)
    wh = np.subtract(np.max(lms, axis=0), xy)

    return np.r_[xy, wh]

class ROIDetector():
    _lower_face = [94, 206, 50, 118, 6, 347, 280, 426, 94]
    RULE = [33, 7, 163, 144, 145, 153, 154, 155 , 133,112,  232, 231, 230, 229, 110 ,33] 
    RARULE = [231, 230, 229, 228, 31, 111, 117, 118, 119, 120, 231] 
    RURE = [263 ,249, 390, 373, 374, 380, 381, 382, 362,452, 451,449, 339, 263]
    RARURE = [451, 450, 449, 448, 261, 340, 346,347, 348, 349, 451]
    def __init__(self, smooth_factor=0.0, **kwargs):
        self.oldroi = None
        self.smooth_factor = smooth_factor
        self.arr = []
        super().__init__(**kwargs)

    def detect(self, frame, arr):
        raise NotImplementedError("detect method needs to be overwritten.")

    def get_roi(self, frame, roi):
        if roi == "forehead" :
            roi = self.detect(frame, self._lower_face)
        elif roi == "RULE" :
            roi = self.detect(frame, self.RULE)
        elif roi == "RARULE" :
            roi = self.detect(frame, self.RARULE)
        elif roi == "RURE" :
            roi = self.detect(frame, self.RURE)
        elif roi == "RARURE" :
            roi = self.detect(frame, self.RARURE)
        return roi
        # self.oldroi = exponential_smooth(roi, self.oldroi, self.smooth_factor)

        # return self.oldroi

    def __call__(self, frame, roi):
        return self.get_roi(frame, roi)
class NoDetector(ROIDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame):
        h, w = frame.shape[:2]
        return RegionOfInterest.from_rectangle(frame, (0, 0), (h, w))


def get_facemesh_coords(landmark_list, frame):
    h, w = frame.shape[:2]
    xys = [(landmark.x, landmark.y) for landmark in landmark_list.landmark]

    return np.multiply(xys, [w, h]).astype(int)

class FaceMeshDetector(ROIDetector):
    #_lower_face =  [67,69,66,107,9,336,296,299,297,338,10,109]
    # _lower_face = [118,119,100,126,209,129,203,206,205,50,118] #check1
    # _lower_face = [355,329,348,347,280,425,426,423,358,429,355] #check2
    #_lower_face = [69,108,151,337,299,296,336,9,107,66,69]
    _lower_face = [94, 206, 50, 118, 6, 347, 280, 426, 94]
    RULE = [33, 7, 163, 144, 145, 153, 154, 155 , 133,112,  232, 231, 230, 229, 110 ,33] 
    RARULE = [231, 230, 229, 228, 31, 111, 117, 118, 119, 120, 231] 
    RURE = [263 ,249, 390, 373, 374, 380, 381, 382, 362,452, 451,449, 339, 263]
    RARURE = [451, 450, 449, 448, 261, 340, 346,347, 348, 349, 451]
    def __init__(self, draw_landmarks=False, refine=False, **kwargs):
        super().__init__(**kwargs)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw_landmarks=draw_landmarks

    def __del__(self):
        self.face_mesh.close()

    def detect(self, frame, arr):
        rawimg = frame.copy()

        frame.flags.writeable = False
        results = self.face_mesh.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks is None:
            return RegionOfInterest(frame, mask=None)

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.multi_face_landmarks,
                               tesselate=True)

        landmarks = get_facemesh_coords(results.multi_face_landmarks[0], frame)
        facerect = get_boundingbox_from_landmarks(landmarks)
        bgmask = get_default_bgmask(frame.shape[1], frame.shape[0])

        return RegionOfInterest.from_contour(rawimg, landmarks[arr],
                                             facerect=facerect, bgmask=bgmask)

    def draw_facemesh(self, img, multi_face_landmarks, tesselate=False,
                      contour=False, irises=False):
        if multi_face_landmarks is None:
            return

        for face_landmarks in multi_face_landmarks:
            if tesselate:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            if contour:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
            if irises and len(face_landmarks) > 468:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
