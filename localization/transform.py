#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 01:49:20 2023

@author: chegde
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
import re
import math


class Transforms():
    
    def __init__(self):
        pass
    
    def translate(self, keypoints):
        """ Translate face keypoints to floor position
        so that perspective transform works correctly on them """
        
        avg_feet = self.extract_avg_feet_positions(keypoints)
        
        translated_keypoints = np.zeros((len(keypoints), 5, 2))
        for i in range(len(keypoints)):
            nose = keypoints[i][0]
            Leye = keypoints[i][1]
            Reye = keypoints[i][2]
            Lear = keypoints[i][3]
            Rear = keypoints[i][4]
            feet = avg_feet[i]
            
            xdiff = feet[0] - nose[0] 
            ydiff = feet[1] - nose[1] 
            
            translated_keypoints[i][0] = (nose[0] + xdiff, nose[1] + ydiff)
            translated_keypoints[i][1] = (Leye[0] + xdiff, Leye[1] + ydiff)
            translated_keypoints[i][2] = (Reye[0] + xdiff, Reye[1] + ydiff)
            translated_keypoints[i][3] = (Lear[0] + xdiff, Lear[1] + ydiff)
            translated_keypoints[i][4] = (Rear[0] + xdiff, Rear[1] + ydiff)
            
        return translated_keypoints


    def extract_avg_feet_positions(self, keypoints):
        """
        This function extracts the average feet positions of all people from keypoints stored by posenet.
        It returns the average of position of left and right feet for each individual.
        
        Inputs:
            keypoints - 10x17x2 matrix containing the x,y coordinates of all keypoints 17 for maximum 10 people in the frame.
                        This matrix corresponds to one frame.
            
        Returns:
            avg_feet_positions - 10x2 array of x,y coordinates of average of left and right foot positions for maximum
                                 of 10 people in the frame.
        """
        
        footX = []
        footY = []
        avg_feet_positions = []
        
        for i in range(len(keypoints)):
            foot_X_temp = [] # Stores avg X positions of feet for all people in one frame 
            foot_Y_temp = [] # Stores avg Y positions of feet for all people in one frame 
            foot_temp = [] # Stores avg x,y positions of feet for all people in one frame 
            
            for j in range(len(keypoints[i])):
                if (j+1)%17 == 0:
                    foot_right_X = keypoints[i][j][0]
                    foot_right_Y = keypoints[i][j][1]
                    foot_left_X = keypoints[i][j-1][0]
                    foot_left_Y = keypoints[i][j-1][1]
                    foot_X_temp.append(int((foot_right_X + foot_left_X)/2))
                    foot_Y_temp.append(int((foot_right_Y + foot_left_Y)/2))
                    foot_temp.append((int((foot_right_X + foot_left_X)/2), int((foot_right_Y + foot_left_Y)/2), 1))
            footX.append(foot_X_temp)
            footY.append(foot_Y_temp)
            avg_feet_positions.append(np.array(foot_temp))
        
        avg_feet_positions = np.array(avg_feet_positions)
        avg_feet_positions = avg_feet_positions.reshape((len(keypoints),3))
        
        return avg_feet_positions
    
    
    def ep6_transform(self, keypoints, source_coordinates, destination_coordinates):
        """ Transform keypoints to top-down view """
    
        keypoints = np.pad(keypoints, ((0,0), (0,0), (0,1)), mode='constant', constant_values=1)
        keypoints_transformed = np.zeros_like(keypoints) # Initialize transformed keypoints
        
        source_coordinates = source_coordinates.astype('float32')
        destination_coordinates = destination_coordinates.astype('float32')

        M = cv2.getPerspectiveTransform(source_coordinates, destination_coordinates)

        for i in range(len(keypoints)):
            for k in range(len(keypoints[i])):
                x = keypoints[i][k][1]
                y = keypoints[i][k][0]
                z = keypoints[i][k][2]

                warped = M.dot([x,y,z])
                warped /= warped[2]
                
                keypoints_transformed[i][k][0] = warped[0]
                keypoints_transformed[i][k][1] = warped[1]
                keypoints_transformed[i][k][2] = warped[2]
                
        return keypoints_transformed
        
    
    def ep6_transform_feet(self, avg_feet_position, source_coordinates, destination_coordinates):
        """ Transform only feet to top-down view """
        
        source_coordinates = source_coordinates.astype('float32')
        destination_coordinates = destination_coordinates.astype('float32')
    
        M = cv2.getPerspectiveTransform(source_coordinates, destination_coordinates)
    
        transformed_positions = []
        for i in range(len(avg_feet_position)):
            x = avg_feet_position[i][1]
            y = avg_feet_position[i][0]
            z = avg_feet_position[i][2]
    
            warped = M.dot([x,y,z])
            warped /= warped[2]
            transformed_positions.append(warped)
    
        transformed_positions = np.array(transformed_positions)
        
        return transformed_positions
    
    def face_to_nose_vector(self, face_keypoints):
        """ Get average positon of all face keypoints other than nose so that vector from this
        average point to nose can be found """
    
        vector = np.zeros((len(face_keypoints), 4)) # [x_avg, y_avg, x_nose, y_nose] for each person
        for i in range(len(face_keypoints)):
            x_avg = face_keypoints[i][1][0] + face_keypoints[i][2][0] + face_keypoints[i][3][0] + face_keypoints[i][4][0]
            y_avg = face_keypoints[i][1][1] + face_keypoints[i][2][1] + face_keypoints[i][3][1] + face_keypoints[i][4][1]
            x_avg /= 4
            y_avg /= 4
            
            vector[i] = np.array((x_avg, y_avg, face_keypoints[i][0][0], face_keypoints[i][0][1]))
            
        return vector
    
    def transform_coordinates(self, pi_ip):
        """ Returns hard coded transform coordinates for selected pi """
        
        if pi_ip == '101':
            source_coordinates = np.array([(286,234), (375,232), (457,456), (204,406)])
            ep6_destination = np.array([(465,148), (447,146), (446,87), (466,94)])
        
        elif pi_ip == '104':
            source_coordinates = np.array([(235,171), (360,168), (510,369), (170,369)])
            ep6_destination = np.array([(173,375), (174,346), (245,345), (244,374)])
            
        elif pi_ip == '106':
            source_coordinates = np.array([(202,179), (298,151), (338,160), (577,360)])
            ep6_destination = np.array([(162,346), (142,376), (134,375), (111,305)])
            
        elif pi_ip == '107':
            source_coordinates = np.array([(249,190), (400,190), (520,270), (210,280)])
            ep6_destination = np.array([(60,215), (105,215), (105,295), (60,295)])
    
        elif pi_ip == '108':
            source_coordinates = np.array([(320,60), (350,60), (445,500), (210,500)])
            ep6_destination = np.array([(267,163), (266,147), (441,146), (442,161)])
            
        elif pi_ip == '114':
            source_coordinates = np.array([(152,176), (387,134), (444,186), (120,303)])
            ep6_destination = np.array([(462,360), (415,374), (416,346), (461,322)])

        elif pi_ip == '115':
            source_coordinates = np.array([(309,20), (351,16), (470,650), (180,650)])
            ep6_destination = np.array([(445,146), (444,160), (267,160), (270,146)])

        elif pi_ip == '118':
            source_coordinates = np.array([(228,209), (347,182), (577,222), (620,257)])
            ep6_destination = np.array([(402,298), (416,267), (460,265), (467,284)])

        elif pi_ip == '120':
            source_coordinates = np.array([(279,164), (388,205), (390,230), (134,430)])
            ep6_destination = np.array([(264,147), (247,125), (247,120), (266,86)])

        elif pi_ip == '123':
            source_coordinates = np.array([(254,59), (445,27), (510,84), (180,257)])
            ep6_destination = np.array([(239,63), (241,118), (173,120), (151,63)])

        elif pi_ip == '124':
            source_coordinates = np.array([(279,164), (388,205), (390,230), (134,430)])
            ep6_destination = np.array([(264,147), (247,125), (247,120), (266,86)])

        elif pi_ip == '125':
            source_coordinates = np.array([(144,249), (348,186), (522,202), (545,359)])
            ep6_destination = np.array([(247,300), (267,267), (298,268), (308,305)])

        elif pi_ip == '129':
            source_coordinates = np.array([(226,113), (333,91), (420,87), (581,140)])
            ep6_destination = np.array([(557,287), (546,267), (546,255), (575,241)])

        elif pi_ip == '132':
            source_coordinates = np.array([(188,144), (259,147), (400,670), (40,670)])
            ep6_destination = np.array([(447,184), (465,183), (465,266), (447,266)])

        elif pi_ip == '133':
            source_coordinates = np.array([(335,31), (440,30), (517,500), (140,570)])
            ep6_destination = np.array([(545,267), (543,284), (467,283), (460,267)])

        elif pi_ip == '135':
            source_coordinates = np.array([(57,272), (240,174), (534,42), (509,312)])
            ep6_destination = np.array([(293,282), (293,293), (290,346), (271,292)])

        elif pi_ip == '136':
            source_coordinates = np.array([(39,403), (229,285), (299,300), (555,302)])
            ep6_destination = np.array([(111,126), (173,125), (160,146), (160,195)])

        elif pi_ip == '137':
            source_coordinates = np.array([(325,41), (430,40), (507,510), (130,580)])
            ep6_destination = np.array([(545,267), (543,284), (467,283), (460,267)])
  
        elif pi_ip == '140':
            source_coordinates = np.array([(79,71), (356,92), (508,150), (630,463)])
            ep6_destination = np.array([(549,339), (582,304), (609,305), (640,340)])
      
        elif pi_ip == '143':
            source_coordinates = np.array([(53,61), (109,44),(316,56), (380,217)])
            ep6_destination = np.array([(297,265), (314,265),(339,298), (289,346)])

        elif pi_ip == '144':
            source_coordinates = np.array([(35,192), (282,137), (360,247), (32,394)])
            ep6_destination = np.array([(538,102), (562,101), (563,122), (539,127)])

        elif pi_ip == '145':
            source_coordinates = np.array([(228,161), (270,143), (512,463), (198,185)])
            ep6_destination = np.array([(357,85), (341,78), (452,47), (377,85)])

        elif pi_ip == '146':
            source_coordinates = np.array([(167,245), (206,200), (403,210), (549,401)])
            ep6_destination = np.array([(57,89), (57,65), (119,66), (105,125)])

        elif pi_ip == '147':
            source_coordinates = np.array([(72,202), (350,107), (382,143), (214,477)])
            ep6_destination = np.array([(338,113), (301,113), (305,102), (341,78)])

        elif pi_ip == '148':
            # source_coordinates = np.array([(180,432), (360,97), (411,84), (434,86)])
            # ep6_destination = np.array([(161,194), (160,345), (143,375), (131,376)])
            source_coordinates = np.array([(332,137), (521,154), (640,309), (151,439)])
            ep6_destination = np.array([(258,532), (170,533), (171,374), (257,347)])

        elif pi_ip == '149':
            source_coordinates = np.array([(228,72), (461,38), (517,127), (578,286)])
            ep6_destination = np.array([(234,31), (300,31), (266,86), (241,119)])

        elif pi_ip == '150':
            source_coordinates = np.array([(191,99), (556,102), (571,133), (173,123)])
            ep6_destination = np.array([(59,209), (105,209), (105,215), (59,215)])
    
        elif pi_ip == '151':
            source_coordinates = np.array([(204,79), (383,76), (498,205), (156,132)])
            ep6_destination = np.array([(160,193), (111,209), (111,121), (160,145)])

        elif pi_ip == '152':
            source_coordinates = np.array([(479,469), (253,394), (168,312), (393,272)])
            ep6_destination = np.array([(317,84), (306,101), (290,110), (283,86)])

        elif pi_ip == '153':
            source_coordinates = np.array([(186,55), (451,57), (600,211), (563,399)])
            ep6_destination = np.array([(116,31), (174,31), (173,121), (161,148)])

        elif pi_ip == '154':
            source_coordinates = np.array([(13,62), (101,52), (406,143), (341,159)])
            ep6_destination = np.array([(60,91), (58,66), (239,48), (239,63)])

        elif pi_ip == '155':
            source_coordinates = np.array([(218,268), (224,237), (408,333), (236,448)])
            ep6_destination = np.array([(333,101), (347,110), (298,102), (297,77)])

        elif pi_ip == '156':
            source_coordinates = np.array([(317,115), (471,105), (521,416), (145,400)])
            ep6_destination = np.array([(333,85), (333,101), (297,101), (297,77)])

        elif pi_ip == '157':
            source_coordinates = np.array([(165,255), (178,16), (536,49), (535,270)])
            ep6_destination = np.array([(576,211), (618,200), (618,240), (575,240)])

        elif pi_ip == '158':
            source_coordinates = np.array([(282,195), (346,215), (352,228), (221,321)])
            ep6_destination = np.array([(246,126), (265,147), (265,162), (246,213)])

        elif pi_ip == '159':
            source_coordinates = np.array([(113,112),(235,126),(553,269),(411,242)])
            ep6_destination = np.array([(476,376), (469,361), (478,317), (481,327)])
 
        elif pi_ip == '160':
            source_coordinates = np.array([(253,130), (574,160), (416,306), (141,222)])
            ep6_destination = np.array([(546,194), (539,137), (577,150), (577,175)])

        elif pi_ip == '161':
            source_coordinates = np.array([(35,368), (197,201), (345,207), (461,392)])
            ep6_destination = np.array([(372,298), (365,324), (347,324), (340,298)])

        elif pi_ip == '162':
            source_coordinates = np.array([(219,14), (305,17), (359,359), (149,353)])
            ep6_destination = np.array([(476,91), (476,77), (537,77), (537,91)])

        elif pi_ip == '163':
            source_coordinates = np.array([(259,65), (300,47), (405,94), (354,124)])
            ep6_destination = np.array([(527,315), (534,315), (534,335), (527,335)])
            
        return source_coordinates, ep6_destination

    
class Plotting():
    
    xs = np.arange(77, 1129, 17.5)
    ys = np.arange(90, 669, 17.5)
    
    def __init__(self):
        pass
    
    def plot_kp_original(self, keypoints, frame, only_face=True, flip_frame=False):
        """ Plot keypoints on EP6 map or image frame """
        
        if only_face == True:
            connection = [[0,1], [0,2], [0,3], 
                          [0,4], [1,3], [2,4]]   
        else:
            connection = [[0,1], [0,2], [0,3], 
                          [0,4], [1,3], [2,4],
                          [5,6], [5,7], [7,9],
                          [6,8], [8,10], [5,11],
                          [6,12], [11,12], [11,13],
                          [13,15], [12,14], [14,16]]
            
        if flip_frame == True:
            frame = np.fliplr(frame)
        
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(frame)
        for i in range(len(keypoints)):
            kp = keypoints[i]
            for j1, j2 in connection:
                
                y1 = kp[j1][0]
                x1 = kp[j1][1]
                y2 = kp[j2][0]
                x2 = kp[j2][1]
                
                if j1 == 0:
                    color1 = 'b'
                else:
                    color1 = 'r'
                    
                ax.plot(x1, y1, 'o', color=color1, markersize=3)
                ax.plot(x2, y2, 'o', color=color, markersize=3)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)
        plt.show()
        
    def plot_original_feet(self, keypoints, frame):
        """ Plot keypoints on EP6 map or image frame """
        
        T = Transforms()
        feet = T.extract_avg_feet_positions(keypoints)
        
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(frame)
        for i in range(len(feet)):
            x = feet[i][1]
            y = feet[i][0]
            
            ax.plot(x, y, 'o', color=color, markersize=3)
        plt.show()
    

    def plot_kp_transform(self, keypoints, avg_feet, frame, plot_feet=False, flip_frame=False):
        """ Plot top-down transformed keypoints on EP6 map or image frame """
        
        connection = [[0,1], [0,2], [0,3], 
                        [0,4], [1,3], [2,4]]  
        
        if flip_frame == True:
            frame = np.fliplr(frame)
        
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(frame)
        for i in range(len(keypoints)):
            kp = keypoints[i]
            feet = avg_feet[i]
            for j1, j2 in connection:
                
                y1 = kp[j1][0]
                x1 = kp[j1][1]
                y2 = kp[j2][0]
                x2 = kp[j2][1]
        
                if j1 == 0:
                    ax.plot(x1, y1, 'o', color='g', markersize=3)
                else:
                    ax.plot(x1, y1, 'o', color=color, markersize=3)
                ax.plot(x2, y2, 'o', color=color, markersize=3)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1)
                
            if plot_feet == True:
                ax.plot(feet[0], feet[1], 'o', color='b', markersize=3)
        plt.show()
    
    def plot_orientation(self, orientation_vectors, floor):
        """ Plot orientation vector pointing from average face keypoint
        position to nose for top-down transformed keypoints. """
    
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(floor)
        for i in range(len(orientation_vectors)):
            
            x = orientation_vectors[i][0]
            y = orientation_vectors[i][1]
            x_nose = orientation_vectors[i][2]
            y_nose = orientation_vectors[i][3]
            
            dx = x_nose - x
            dy = y_nose - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x_nose, y_nose, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()

    def plot_orientation_and_gt(self, orientation_vectors, floor, frame):
        """ Display orientations estimated and ground truth images next
        to each other """
        
        color = 'r'
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
        ax1.imshow(floor)
        for i in range(len(orientation_vectors)):
            
            x = orientation_vectors[i][0]
            y = orientation_vectors[i][1]
            x_nose = orientation_vectors[i][2]
            y_nose = orientation_vectors[i][3]
            
            dx = x_nose - x
            dy = y_nose - y
            
            #ax1.plot(x, y, 'o', color=color, markersize=3)
            #ax1.plot(x_nose, y_nose, 'o', color=color, markersize=3)
            ax1.arrow(x, y, dx, dy, color=color, width=1)
            
        ax2.imshow(frame)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(wspace=0, hspace=0)    
        #plt.show()
        
        return fig
    
    def plot_orientation3D(self, orientation_vectors, positions, floor):
        """ Plot orientation vector pointing from average face keypoint
        position to nose for top-down transformed keypoints. """
    
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(floor)
        for i in range(len(orientation_vectors)):
            
            x = positions[i][0]
            y = positions[i][1]
            r = 1
            
            x1 = x + r*math.cos(math.radians(orientation_vectors[i]))
            y1 = y + r*math.sin(math.radians(orientation_vectors[i]))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()
        
    def plot_orientation2D3D(self, orientations2D, orientations3D, positions, floor):
        """ Plot orientation vector for 2D and 3D orientations. """
        
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(floor)
        
        for i in range(len(orientations2D)):
            
            x = orientations2D[i][0]
            y = orientations2D[i][1]
            x_nose = orientations2D[i][2]
            y_nose = orientations2D[i][3]
            
            dx = x_nose - x
            dy = y_nose - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x_nose, y_nose, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
            
        color = 'b'
        for i in range(len(orientations3D)):
            
            x = positions[i][0]
            y = positions[i][1]
            r = 1
            
            x1 = x + r*math.cos(math.radians(360-orientations3D[i]))
            y1 = y + r*math.sin(math.radians(360-orientations3D[i]))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()
        
    def plot_orientations(self, orientations, positions, floor, figure, color):
        
        #fig, ax = plt.subplots()
        fig, ax = figure
        ax.imshow(floor)
        
        for i in range(len(orientations)):
            
            x = positions[i][0]
            y = positions[i][1]
            r = 1
            
            x1 = x + r*math.cos(math.radians(360-orientations[i]))
            y1 = y + r*math.sin(math.radians(360-orientations[i]))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()
        
    def plot_angles(self, positions, orientations, floor):
        
        color = 'r'
        fig, ax = plt.subplots()
        ax.imshow(floor)
        
        r = 1
        for i in range(len(orientations)):
            
            x = positions[i][0]
            y = positions[i][1]
            
            x1 = x + r*math.cos(math.radians(orientations[i]))
            y1 = y + r*math.sin(math.radians(orientations[i]))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()
        
    def plot_pos_ori_rescaled(self, orientations, positions, floor, figure, color):
        
        #fig, ax = plt.subplots()
        fig, ax = figure
        ax.imshow(floor)
        
        for i in range(len(orientations)):
            
            x = positions[i][0]
            y = positions[i][1]
            x = self.xs[int(x)]
            y = self.ys[34-int(y)-1]
            #x = 17.5*x
            #y = 17.5*y
            r = 8
            
            x1 = x + r*math.cos(math.radians(360-orientations[i]))
            y1 = y + r*math.sin(math.radians(360-orientations[i]))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()
        
    def plot_hyeok(self, orientations, positions, floor, figure, color):
        
        fig, ax = figure
        ax.imshow(floor)
        
        for i in range(len(orientations)):
            
            x = positions[i][0]
            y = positions[i][1]
            # x = self.xs[int(x)]
            # y = self.ys[34-int(y)-1]
            #x = 17.5*x
            #y = 17.5*y
            r = 8
            
            x1 = x + r*math.cos(math.radians(orientations[i]-90))
            y1 = y + r*math.sin(math.radians(orientations[i]-90))
            
            dx = x1 - x
            dy = y1 - y
            
            ax.plot(x, y, 'o', color=color, markersize=3)
            ax.plot(x1, y1, 'o', color=color, markersize=3)
            ax.arrow(x, y, dx, dy, color=color, width=1)
        plt.show()       

class Utils():
    
    def __init__(self):
        pass
    
    def get_pi_ip(self, frame_path):
        """ Get IP address info from pi video path """
        
        _, year, month, day, hour, pi_ip, _, _, time = re.findall(r"\d+", frame_path)
        
        # year = date[0:4]
        # month = date[4:6]
        # day = date[6::]
        
        # hour = time[0:2]
        minute = time[2:4]
        second = time[4::]
        
        return pi_ip, year, month, day, hour, minute, second
    
    def remove_fp(self, keypoints_minute):
        """ Naive way of removing false positive skeleton detections """
        
        cleaned_keypoints = deepcopy(keypoints_minute)
        keys = list(keypoints_minute.keys())
        for i in range(len(keys)-1):
            if keys[i] != '__global__' or keys[i] != '__header__' or keys[i] != '__version__':
                kp1 = keypoints_minute[keys[i]]
                kp2 = keypoints_minute[keys[i+1]]
                for j in range(len(kp1)):
                    k1 = kp1[j]
                    keep = True
                    for k in range(len(kp2)):
                        if np.all(k1 == kp2[k]):
                            keep = False
                            break
                    if keep == False:
                        cleaned_keypoints[keys[i]][j] = np.zeros((17,2))
                        if i == len(keys) - 1:
                            cleaned_keypoints[keys[i+1]][j] = np.zeros((17,2))
                            
        return cleaned_keypoints
    
    def convert_angle_to_degrees(self, orientations):
        
        """ Convert angles from vector notation to degree notation """
        
        orientations_degrees = np.zeros((len(orientations),1))
        
        for i in range(len(orientations)):
            # x1, y1 is point of origin of vector
            x1, y1, x2, y2 = orientations[i]
            y = y2 - y1
            x = x2 - x1
            
            #theta = math.degrees(math.atan2(y,x))
            theta = math.degrees(np.arctan2(y,x))
            
            theta = -1*theta
            if theta < 0:
                theta += 360
            #theta = 360 - theta # Uncomment to get angles in image coordinate system (i.e., (0,0) at top left)
            orientations_degrees[i] = theta
            
        return orientations_degrees
                