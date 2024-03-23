# import the necessary packages
import os
import cv2
import time
import mediapipe
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

visualize = False


class EyeDetection():
    def __init__(self) -> None:
        # frames the eye must be below the threshold
        self.EYE_AR_THRESH = 0.45

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
        self.chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def get_eye_state(self, landmarks):
        leftEye = landmarks[self.chosen_left_eye_idxs]
        rightEye = landmarks[self.chosen_right_eye_idxs]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR)

        return leftEye, rightEye, ear, self.check_ear(ear)

    def check_ear(self, ear):
        if ear < self.EYE_AR_THRESH:
            return False
        return True

    def run(self, frame, if_visual=False):
        is_eye_open = False

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye, rightEye, ear, is_eye_open = self.get_eye_state(shape)

            if if_visual:
                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if not is_eye_open:
                if if_visual:
                    cv2.putText(frame, "Eye: {}".format("close"), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                if if_visual:
                    cv2.putText(frame, "Eye: {}".format("Open"), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # only for single person
            break

        return is_eye_open, frame


if __name__ == '__main__':
    vs = VideoStream(src=0).start()

    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_eye.avi', fourcc, 20.0, (400, 400), True)
        # vs = VideoStream(usePiCamera=True).start()

    eye_detect = EyeDetection()
    time.sleep(1.0)

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        is_eye_open, frame = eye_detect.run(frame, if_visual=visualize)

        if visualize:
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame

            # show the frame
            frame = cv2.resize(frame, (400, 400))
            cv2.imshow("Frame", frame)
            out.write(frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    vs.stop()
    if visualize:
        cv2.destroyAllWindows()
        out.release()
