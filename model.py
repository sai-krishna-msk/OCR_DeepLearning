
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import imutils
import cv2
import pickle
from dl import Test
import os

final =[]

def main(path):

	def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):

		charIter = charCnts.__iter__()
		rois = []
		locs = []


		while True:
			try:

				c = next(charIter)
				(cX, cY, cW, cH) = cv2.boundingRect(c)
				roi = None


				if cW >= minW and cH >= minH:
					roi = image[cY:cY + cH, cX:cX + cW]
					rois.append(roi)
					locs.append((cX, cY, cX + cW, cY + cH))

				else:

					parts = [c, next(charIter), next(charIter)]
					(sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
						-np.inf)

					for p in parts:

						(pX, pY, pW, pH) = cv2.boundingRect(p)
						sXA = min(sXA, pX)
						sYA = min(sYA, pY)
						sXB = max(sXB, pX + pW)
						sYB = max(sYB, pY + pH)

					roi = image[sYA:sYB, sXA:sXB]
					rois.append(roi)
					locs.append((sXA, sYA, sXB, sYB))


			except StopIteration:
				break

		return (rois, locs)
	chars = {}
	charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
		"T", "U", "A", "D"]



	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
	output = []


	image = cv2.imread(path)
	(h, w,) = image.shape[:2]
	delta = int(h - (h * 0.2))
	bottom = image[delta:h, 0:w]
	cv2.imshow("ROI" , bottom)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



	gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

	cv2.imshow("Morphology" , blackhat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
		ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")


	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

	thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	thresh = clear_border(thresh)

	cv2.imshow("threshold" , thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



	groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
	groupLocs = []

	for (i, c) in enumerate(groupCnts):
		(x, y, w, h) = cv2.boundingRect(c)


		if w > 50 and h > 15:
			groupLocs.append((x, y, w, h))

	groupLocs = sorted(groupLocs, key=lambda x:x[0])

	for (gX, gY, gW, gH) in groupLocs:
		groupOutput = []


		group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
		group = cv2.threshold(group, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]




		charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		charCnts = charCnts[0] if imutils.is_cv2() else charCnts[1]
		charCnts = contours.sort_contours(charCnts,
			method="left-to-right")[0]

		(rois, locs) = extract_digits_and_symbols(group, charCnts)

		for roi in rois:


			scores = []
			roi = cv2.resize(roi, (36, 36))
			cv2.imwrite("test/test.jpg" , roi)
			currentResult = Test("test/test.jpg")
			final.append(currentResult)


	os.remove("test/test.jpg")
	return  final


sai = main("check_image/check.jpg")
print(sai)
