import numpy as np, pandas as pd, cv2, os, re

path_in = "../ims"
path_out = "./results.csv"
sigma = 0.33 # percentage upper/lowerbound for canny
conversion_factor = 0.474687961 # um/px
dmin = 20 # minimum beadd iameter in um
rmin = int(np.round(dmin/conversion_factor/2, 0))
rdmin = 2*rmin#minimum distance between centerpoints
dmax = 60 # maximal bead diameter
rmax = int(np.round(dmax/conversion_factor/2, 0))

def bhb_diam(im, rmin, rmax, rdmin):
    rerun = 1
    while rerun =  = 1:
        eq = cv2.equalizeHist(im) # equalize histogram to improve contrast
        blurred = cv2.GaussianBlur(eq, (3, 3), 0) # gaussian blur to smoothe
        v = np.median(blurred) # take median intensity of blurred image
        lower = int(max(0, (1.0-sigma)*v)) # lower bound for Canny filter
        upper = int(min(255, (1.0+sigma)*v)) # upper bound for Canny filter
        edged = cv2.Canny(blurred, lower, upper) # apply Canny filter to detect edges
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, rdmin, param1 = 30, param2 = 30, minRadius = rmin, maxRadius = rmax)
        nbeads = int(np.round(np.divide(circles.size, 3), 2)) # calculate total number of beads
        dmean = np.round((np.mean(circles, axis = 1)[0, 2]*conversion_factor*2), 2)
        dstd = np.round((np.std(circles, axis = 1)[0, 2]*conversion_factor*2), 2)
        reldstd = np.round(np.multiply(np.divide(dstd, dmean), 100), 2)
        im = cv2.fastNlMeansDenoising(im, 10, 10, 9, 25)

# apply the function to all.tiff files in path_in
total_results = np.empty((0, 11))
for file in os.listdir(path_in):
    if file.endswith(".tif"):
        path_im = path_in+"/"+file
        im = cv2.imread(path_im, -1)
        output, nbeads, dmean, dstd, reldstd = bhb_diam(im, rmin, rmax, rdmin)

        settings = re.split('_', file)
        settings2 = re.split('-', settings[3])
        r = int(settings[2])/int(settings[1])
        Q = (int(settings[2])+int(settings[1]))/1000]
        im_results = np.array([settings[0], settings[1], settings[2], r, Q, settings2[0], re.sub(".tif", "", settings2[1]), nbeads, dmean, dstd, reldstd])
        total_results = np.append(total_results, [im_results], axis = 0)
        cv2.imwrite(str(re.sub(".tif", "", file)+"_processed.tif"), output)

cols = np.array(['chem', 'ap', 'oil', 'r', 'Q', 's', 'pic', 'n', 'dmean[um]', 'dstd', 'reldstd'])
df = pd.DataFrame(data = total_results, columns = cols)
df_sorted = df.sort_values(by = ['s', 'pic'])
print(df_sorted)
