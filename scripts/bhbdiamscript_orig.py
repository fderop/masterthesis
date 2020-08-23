import numpy as np
import pandas as pd
import cv2
import os
import re

path_in = "../ims"
path_out = "./results.csv"
sigma = 0.33 # percentage upper/lower bound for canny
conversion_factor = 0.474687961 # µm/px
dmin = 20 # minimum bead diameter in µm
rmin = int(np.round(dmin / conversion_factor / 2, 0)) # convert µm to px, diam to radius
rdmin = 2 * rmin # minimum distance between centerpoints of beads
dmax = 60 # maximal bead radius
rmax = int(np.round(dmax / conversion_factor / 2, 0)) # convert µm to px, diam to radius

def bhb_diam(im, rmin, rmax, rdmin):
    rerun = 1
    while rerun == 1:
        eq = cv2.equalizeHist(im) # equalize histogram to improve contrast

        blurred = cv2.GaussianBlur(eq, (3, 3), 0) # gaussian blur to smoothe

        # calculate settings for Canny fillter
        v = np.median(blurred) # take median intensity of blurred image
        lower = int(max(0, (1.0 - sigma) * v)) # lower bound for Canny filter
        upper = int(min(255, (1.0 + sigma) * v)) # upper bound for Canny filter
        edged = cv2.Canny(blurred,lower,upper) # apply Canny filter to detect edges

        # apply the Hough circle finding algorithm on the edges
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, rdmin,
                                   param1 = 30, param2 = 30, minRadius = rmin, maxRadius = rmax)

        nbeads = int(np.round(np.divide(circles.size,3),2)) # calculate total number of beads
        dmean = np.round((np.mean(circles, axis = 1)[0,2] * conversion_factor * 2),2) # calculate mean diameter [µm]
        dstd = np.round((np.std(circles, axis = 1)[0,2] * conversion_factor * 2),2) # calculate diameter standard deviation
        reldstd = np.round(np.multiply(np.divide(dstd,dmean),100),2) # calculate relative diameter standard deviation

        if nbeads < 250: # arbitrary number 250: if the script detects more than 250 beads, we assume something went wrong and
                         # we rerun the script after applying a noise reduction filter to the original image
            rerun = 0
            circles = np.round(circles[0, :]).astype("int") # round the circles array
            output = im.copy() # create a copy of the image to use as background for the overlay
            for (x, y, r) in circles: # create an overlay showing all detected circles on top of the image copy
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            return output, nbeads, dmean, dstd, reldstd
        else:
            im = cv2.fastNlMeansDenoising(im, 10, 10, 9, 25)

total_results = np.empty((0, 11)) # load an empty total_results array
for file in os.listdir(path_in):
    if file.endswith(".tif"): # discard non .tif files
        path_im = path_in + "/" + file # image path = directory path + filename
        im = cv2.imread(path_im,-1) # read the image
        output, nbeads, dmean, dstd, reldstd = bhb_diam(im, rmin, rmax, rdmin) # apply the circle detection function

        settings = re.split('_', file) # read the filename to get experimental settings
        settings2 = re.split('-',settings[3])
        r = int(settings[2]) / int(settings[1]) # ratio of flows
        Q = (int(settings[2]) + int(settings[1])) / 1000 # total flow speed [µL/h]
        im_results = np.array([settings[0], settings[1], settings[2], r, Q, settings2[0],
                               re.sub(".tif","",settings2[1]), nbeads, dmean, dstd, reldstd])
        total_results = np.append(total_results, [im_results], axis = 0) # append image results to total results

        cv2.imwrite(str(re.sub(".tif","",file) + "_processed.tif"), output)

# create a a pd DataFrame and sort it by dmean
cols = np.array(['chem', 'ap', 'oil', 'r', 'Q', 's', 'pic', 'n', 'dmean [µm]', 'dstd', 'reldstd'])
df = pd.DataFrame(data = total_results, columns = cols)
df_sorted = df.sort_values(by=['s','pic'])
print(df_sorted)ath_in = "../ims"
path_out = "./results.csv"
sigma = 0.33 # percentage upper/lower bound for canny
conversion_factor = 0.474687961 # µm/px
dmin = 20 # minimum bead diameter in µm
rmin = int(np.round(dmin / conversion_factor / 2, 0)) # convert µm to px, diam to radius
rdmin = 2 * rmin # minimum distance between centerpoints of beads
dmax = 60 # maximal bead radius
rmax = int(np.round(dmax / conversion_factor / 2, 0)) # convert µm to px, diam to radius

def bhb_diam(im, rmin, rmax, rdmin):
    rerun = 1
    while rerun == 1:
        eq = cv2.equalizeHist(im) # equalize histogram to improve contrast

        blurred = cv2.GaussianBlur(eq, (3, 3), 0) # gaussian blur to smoothe

        # calculate settings for Canny fillter
        v = np.median(blurred) # take median intensity of blurred image
        lower = int(max(0, (1.0 - sigma) * v)) # lower bound for Canny filter
        upper = int(min(255, (1.0 + sigma) * v)) # upper bound for Canny filter
        edged = cv2.Canny(blurred,lower,upper) # apply Canny filter to detect edges

        # apply the Hough circle finding algorithm on the edges
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, rdmin,
                                   param1 = 30, param2 = 30, minRadius = rmin, maxRadius = rmax)

        nbeads = int(np.round(np.divide(circles.size,3),2)) # calculate total number of beads
        dmean = np.round((np.mean(circles, axis = 1)[0,2] * conversion_factor * 2),2) # calculate mean diameter [µm]
        dstd = np.round((np.std(circles, axis = 1)[0,2] * conversion_factor * 2),2) # calculate diameter standard deviation
        reldstd = np.round(np.multiply(np.divide(dstd,dmean),100),2) # calculate relative diameter standard deviation

        if nbeads < 250: # arbitrary number 250: if the script detects more than 250 beads, we assume something went wrong and
                         # we rerun the script after applying a noise reduction filter to the original image
            rerun = 0
            circles = np.round(circles[0, :]).astype("int") # round the circles array
            output = im.copy() # create a copy of the image to use as background for the overlay
            for (x, y, r) in circles: # create an overlay showing all detected circles on top of the image copy
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            return output, nbeads, dmean, dstd, reldstd
        else:
            im = cv2.fastNlMeansDenoising(im, 10, 10, 9, 25)

total_results = np.empty((0, 11)) # load an empty total_results array
for file in os.listdir(path_in):
    if file.endswith(".tif"): # discard non .tif files
        path_im = path_in + "/" + file # image path = directory path + filename
        im = cv2.imread(path_im,-1) # read the image
        output, nbeads, dmean, dstd, reldstd = bhb_diam(im, rmin, rmax, rdmin) # apply the circle detection function

        settings = re.split('_', file) # read the filename to get experimental settings
        settings2 = re.split('-',settings[3])
        r = int(settings[2]) / int(settings[1]) # ratio of flows
        Q = (int(settings[2]) + int(settings[1])) / 1000 # total flow speed [µL/h]
        im_results = np.array([settings[0], settings[1], settings[2], r, Q, settings2[0],
                               re.sub(".tif","",settings2[1]), nbeads, dmean, dstd, reldstd])
        total_results = np.append(total_results, [im_results], axis = 0) # append image results to total results

        cv2.imwrite(str(re.sub(".tif","",file) + "_processed.tif"), output)

# create a a pd DataFrame and sort it by dmean
cols = np.array(['chem', 'ap', 'oil', 'r', 'Q', 's', 'pic', 'n', 'dmean [µm]', 'dstd', 'reldstd'])
df = pd.DataFrame(data = total_results, columns = cols)
df_sorted = df.sort_values(by=['s','pic'])
print(df_sorted)
