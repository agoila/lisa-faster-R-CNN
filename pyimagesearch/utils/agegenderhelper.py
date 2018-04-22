# import the necessary packages
import numpy as np
import glob
import cv2
import os

class AgeGenderHelper:
	def __init__(self, config):
		# store the configuration object and build the age bins used
		# for constructing class labels
		self.config = config
		self.ageBins = self.buildAgeBins()

	def buildAgeBins(self):
		# initialize the list of age bins based on the Adience
		# dataset
		ageBins = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32),
			(38, 43), (48, 53), (60, np.inf)]

		# return the age bins
		return ageBins

	def toLabel(self, age, gender):
		# check to see if we should determine the age label
		if self.config.DATASET_TYPE == "age":
			return self.toAgeLabel(age)

		# otherwise, assume we are determining the gender label
		return self.toGenderLabel(gender)

	def toAgeLabel(self, age):
		# initialize the label
		label = None

		# break the age tuple into integers
		age = age.replace("(", "").replace(")", "").split(", ")
		(ageLower, ageUpper) = np.array(age, dtype="int")

		# loop over the age bins
		for (lower, upper) in self.ageBins:
			# determine if the age falls into the current bin
			if ageLower >= lower and ageUpper <= upper:
				label = "{}_{}".format(lower, upper)
				break

		# return the label
		return label

	def toGenderLabel(self, gender):
		# return 0 if the gender is male, 1 if the gender is female
		return 0 if gender == "m" else 1

	def buildOneOffMappings(self, le):
		# sort the class labels in ascending order (according to age)
		# and initialize the one-off mappings for computing accuracy
		classes = sorted(le.classes_, key=lambda x:
			int(x.decode("utf-8").split("_")[0]))
		oneOff = {}

		# loop over the index and name of the (sorted) class labels
		for (i, name) in enumerate(classes):
			# determine the index of the *current* class label name
			# in the *label encoder* (unordered) list, then
			# initialize the index of the previous and next age
			# groups adjacent to the current label
			current = np.where(le.classes_ == name)[0][0]
			prev = -1
			next = -1

			# check to see if we should compute previous adjacent
			# age group
			if i > 0:
				prev = np.where(le.classes_ == classes[i - 1])[0][0]

			# check to see if we should compute the next adjacent
			# age group
			if i < len(classes) - 1:
				next = np.where(le.classes_ == classes[i + 1])[0][0]

			# construct a tuple that consists of the current age
			# bracket, the previous age bracket, and the next age
			# bracket
			oneOff[current] = (current, prev, next)

		# return the one-off mappings
		return oneOff

	def buildPathsAndLabels(self):
		# initialize the list of image paths and labels
		paths = []
		labels = []

		# grab the paths to the folds files
		foldPaths = os.path.sep.join([self.config.LABELS_PATH,
			"*.txt"])
		foldPaths = glob.glob(foldPaths)

		# loop over the folds paths
		for foldPath in foldPaths:
			# load the contents of the folds file, skipping the
			# header
			rows = open(foldPath).read()
			rows = rows.strip().split("\n")[1:]

			# loop over the rows
			for row in rows:
				# unpack the needed components of the row
				row = row.split("\t")
				(userID, imagePath, faceID, age, gender) = row[:5]

				# if the age or gender is invalid, ignore the sample
				if age[0] != "(" or gender not in ("m", "f"):
					continue

				# construct the path to the input image and build
				# the class label
				p = "landmark_aligned_face.{}.{}".format(faceID,
					imagePath)
				p = os.path.sep.join([self.config.IMAGES_PATH,
					userID, p])
				label = self.toLabel(age, gender)

				# if the label is None, then the age does not fit
				# into our age brackets, ignore the sample
				if label is None:
					continue

				# update the respective image paths and labels lists
				paths.append(p)
				labels.append(label)

		# return a tuple of image paths and labels
		return (paths, labels)

	@staticmethod
	def visualizeAge(agePreds, le):
		# initialize the canvas and sort the predictions according
		# to their probability
		canvas = np.zeros((250, 310, 3), dtype="uint8")
		idxs = np.argsort(agePreds)[::-1]

		# loop over the age predictions in ascending order
		for (i, j) in enumerate(idxs):
			# construct the text for the prediction
			#ageLabel = le.inverse_transform(j) # Python 2.7
			ageLabel = le.inverse_transform(j).decode("utf-8")
			ageLabel = ageLabel.replace("_", "-")
			ageLabel = ageLabel.replace("-inf", "+")
			text = "{}: {:.2f}%".format(ageLabel, agePreds[j] * 100)

			# draw the label + probability bar on the canvas
			w = int(agePreds[j] * 300) + 5
			cv2.rectangle(canvas, (5, (i * 35) + 5),
				(w, (i * 35) + 35), (0, 0, 255), -1)
			cv2.putText(canvas, text, (10, (i * 35) + 23),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
		
		# return the visualization
		return canvas

	@staticmethod
	def visualizeGender(genderPreds, le):
		# initialize the canvas and sort the predictions according
		# to their probability
		canvas = np.zeros((100, 310, 3), dtype="uint8")
		idxs = np.argsort(genderPreds)[::-1]

		# loop over the gender predictions in ascending order
		for (i, j) in enumerate(idxs):
			# construct the text for the prediction
			gender = le.inverse_transform(j)
			gender = "Male" if gender == 0 else "Female"
			text = "{}: {:.2f}%".format(gender, genderPreds[j] * 100)

			# draw the label + probability bar on the canvas
			w = int(genderPreds[j] * 300) + 5
			cv2.rectangle(canvas, (5, (i * 35) + 5),
				(w, (i * 35) + 35), (0, 0, 255), -1)
			cv2.putText(canvas, text, (10, (i * 35) + 23),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

		# return the canvas
		return canvas