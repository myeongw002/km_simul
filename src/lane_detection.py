#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np




class Lane_Detector():
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.img = None
        rospy.init_node("lane_detector")
        rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback)


    def img_callback(self,msg:Image):
        self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        #cv2.imshow("Image window", self.img)
        #cv2.waitKey(1)



    def merge_mask(self, mask1, mask2):
            """Merge two masks by finding the largest contour in mask2 and combining it with mask1 if the centers are close."""
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    center_distance = np.sqrt((cX - mask1.shape[1] / 2) ** 2 + (cY - mask1.shape[0] / 2) ** 2)
                    if center_distance < mask1.shape[0] * 0.1:  # adjust threshold as necessary
                        mask1 = cv2.bitwise_or(mask1, mask2)
            return mask1

    def traffic_light_check(self, cropped_image):
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Using the color ranges to create masks
        color_ranges = {
            'red1': ([0, 100, 100], [10, 255, 255]),
            'red2': ([160, 100, 100], [180, 255, 255]),
            'green': ([40, 50, 50], [90, 255, 255]),
            'yellow': ([15, 50, 50], [35, 255, 255])
        }

        masks = {}
        for color, (lower, upper) in color_ranges.items():
            masks[color] = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Combine the red masks
        masks['red'] = cv2.bitwise_or(masks['red1'], masks['red2'])
        del masks['red1'], masks['red2']

        # Counting the white pixels in each mask and identify the most prominent color
        max_count = 0
        most_prominent_color = None
        for color, mask in masks.items():
            white_pixel_count = np.sum(mask == 255)
            if white_pixel_count > max_count:
                max_count = white_pixel_count
                most_prominent_color = color

            print(f"{color} white pixel count: {white_pixel_count}")
            cv2.imshow(f"{color} mask", mask)

        print(f"Most prominent color: {most_prominent_color}")
        print('-----------------')
        cv2.imshow('Cropped Traffic Light', cropped_image)
        cv2.waitKey(1)

        return most_prominent_color



    def traffic_light(self, img):
        # Get the dimensions of the image
        h, w, _ = img.shape
        # Define the region of interest (ROI)
        roi_w = int(w * 0.5)
        roi_h = int(h * 0.5)
        roi = img[:roi_h, roi_w:w]

        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.medianBlur(gray_roi, 5)

        # Detect circles in the ROI using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        # If circles are detected, draw them and determine their color
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                # Crop the circle region from the image
                xmin = max(0, i[0] - i[2])
                xmax = min(roi_w, i[0] + i[2])
                ymin = max(0, i[1] - i[2])
                ymax = min(roi_h, i[1] + i[2])
                cropped_image = roi[ymin:ymax, xmin:xmax]
                self.traffic_light_check(cropped_image)

                # Draw the outer circle
                cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)
                # Detect the color of the cropped circle region
                

        # Display the ROI with detected circles
        cv2.imshow("ROI window", roi)
        cv2.waitKey(1)



    def main(self):
        rospy.wait_for_message("/usb_cam/image_raw", Image)
        
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            cv2.imshow("Image window", self.img)
            cv2.waitKey(1) 
            
            self.traffic_light(self.img)
            rate.sleep()






if __name__ == '__main__':
    detector = Lane_Detector()
    detector.main()