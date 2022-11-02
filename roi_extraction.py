import cv2
import mediapipe as mp
import numpy as np


class RoIExtraction():
    forehead_roi = 0
    nose_roi = 0
    face_roi = 0
    rule_roi = 0 
    rarule_roi = 0 
    rure_roi = 0 
    rarure_roi = 0
    def __init__(self):

        # Arrays are list of landmark for each RoI location
        #self.forehead_arr = [118,119,100,126,209,129,203,206,205,50,118] 
        #self.face_arr =  [355,329,348,347,280,425,426,423,358,429,355]
        self.forehead_arr = [105, 103, 67, 109, 10, 338, 297, 332, 334, 105]
        self.nose_arr = [94, 206, 50, 118, 6, 347, 280, 426, 94]
        self.face_arr = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 376, 411, 426, 94, 206, 187, 147, 137, 234,127, 162, 21, 54, 103, 67, 109, 10]
        self.RULE = [33, 7, 163, 144, 145, 153, 154, 155 , 133,112,  232, 231, 230, 229, 110 ,33] 
        self.RARULE = [231, 230, 229, 228, 31, 111, 117, 118, 119, 120, 231] 
        self.RURE = [263 ,249, 390, 373, 374, 380, 381, 382, 362,452, 451,449, 339, 263]
        self.RARURE = [451, 450, 449, 448, 261, 340, 346,347, 348, 349, 451]
        self.outline_forehead = []
        self.outline_nose = []
        self.outline_face = []
        self.outline_rule = []
        self.outline_rarule = []
        self.outline_rure = []
        self.outline_rarure = []


    def __call__(self, image):
        self.roi_extraction(image)
    def setRule_roi(self, rule_roi):
        self.rule_roi = rule_roi

    def getRule_roi(self):
        return self.forehead_roi
    def setRaRule_roi(self, rarule_roi):
        self.rarule_roi = rarule_roi

    def getRarule_roi(self):
        return self.rarule_roi
    def setRure_roi(self, rure_roi):
        self.rure_roi = rure_roi

    def getRure_roi(self):
        return self.rure_roi

    def getrarure_roi(self):
        return self.rarure_roi
    def setRaRure_roi(self, rarure_roi):
        self.rarure_roi = rarure_roi

    def setForehead_roi(self, forehead_roi):
        self.forehead_roi = forehead_roi

    def getForehead_roi(self):
        return self.forehead_roi

    def setNose_roi(self, nose_roi):
        self.nose_roi = nose_roi

    def getNose_roi(self):
        return self.nose_roi

    def setFace_roi(self, face_roi):
        self.face_roi = face_roi

    def getFace_roi(self):
        return self.face_roi

    def roi_extraction(self, image):
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(image)
            mask_forehead, self.outline_forehead = self.create_mask(results, self.forehead_arr, image)
            mask_nose, self.outline_nose = self.create_mask(results, self.nose_arr, image)
            mask_face, self.outline_face = self.create_mask(results, self.face_arr, image)
            # mask_rule, self.outline_rule = self.create_mask(results, self.RULE, image)
            # mask_rarule, self.outline_rarule = self.create_mask(results, self.RARULE, image)
            # mask_rure, self.outline_rure = self.create_mask(results, self.RURE, image)
            # mask_rarure, self.outline_rarure = self.create_mask(results, self.RARURE, image)

            self.setForehead_roi(mask_forehead)
            self.setNose_roi(mask_nose)
            self.setFace_roi(mask_face)
            #self.setRule_roi(mask_rule)
            # self.setRaRule_roi(mask_rarule)
            # self.setRure_roi(mask_rure)
            # self.setRaRure_roi(mask_rarure)

            # Extraction RoI from Image using landmarks of mediapipe Face Detection
    def DCES(self, image) :
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(image)
            mask_rule, self.outline_rule = self.create_mask(results, self.RULE, image)
            mask_rarule, self.outline_rarule = self.create_mask(results, self.RARULE, image)
            mask_rure, self.outline_rure = self.create_mask(results, self.RURE, image)
            mask_rarure, self.outline_rarure = self.create_mask(results, self.RARURE, image)
            self.setRule_roi(mask_rule)
            self.setRaRule_roi(mask_rarule)
            self.setRure_roi(mask_rure)
            self.setRaRure_roi(mask_rarure)

    def create_mask(self, results, roi_arr, image):
        outline = []

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                count = 0

                while count < len(roi_arr):
                    temp = roi_arr[count]

                    # Nomarlize landmark to pixels coordinates

                    x = face_landmarks.landmark[temp].x
                    y = face_landmarks.landmark[temp].y
                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    # add pixels coordinates to array

                    outline.append((relative_x, relative_y))
                    count = count + 1

        # Using pixels coordinates array to create mask

        mask = np.zeros((image.shape[0], image.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(outline), 1)
        mask = mask.astype(np.bool_)
        return mask, outline
