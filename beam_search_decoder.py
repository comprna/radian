"""
This has been adapted from https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
"""


from __future__ import division
from __future__ import print_function
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import tensorflow as tf
# from copy import deepcopy

N_BASES = 4

class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal = 0 # blank and non-blank
		self.prNonBlank = 0 # non-blank
		self.prBlank = 0 # blank
		self.prText = 1 # LM score
		self.lmApplied = False # flag if LM was already applied to this beam
		self.labeling = () # beam-labeling
		self.indices = () # indices of each character in beam


class BeamState:
	"information about the beams at specific time-step"
	def __init__(self):
		self.entries = {}

	def norm(self):
		"length-normalise LM score"
		for (k, _) in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

	def sort(self):
		"return beam-labelings, sorted by probability"
		beams = [v for (_, v) in self.entries.items()]
		sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
		return [x.labeling for x in sortedBeams], [x.indices for x in sortedBeams]
		# return [x.labeling for x in sortedBeams], [x.indices for x in sortedBeams], [x.prTotal*x.prText for x in sortedBeams]
		# return [x.labeling for x in sortedBeams]

def applyRNAModel(parentBeam, childBeam, lm, cache, lmFactor, lenContext):
	"calculate RNA model score of child beam by taking score from parent beam and 6-mer probability of last six chars"
	if lm and not childBeam.lmApplied:
		if len(parentBeam.labeling) < lenContext:
			return

		contextTup = parentBeam.labeling[-lenContext:] # context tuple
		if contextTup not in cache:
			context = tf.one_hot(list(parentBeam.labeling[-lenContext:]), N_BASES).numpy()
			context = context.reshape(1, lenContext, N_BASES)
			probs = lm.predict(context)
			cache[contextTup] = probs
		else:
			probs = cache[contextTup]

		newChar = childBeam.labeling[-1]
		probNewChar = probs[0][newChar]

		probNewChar = probNewChar ** lmFactor # probability of seeing k-mer
		childBeam.prText = parentBeam.prText * probNewChar # probability of whole sequence
		childBeam.lmApplied = True # only apply LM once per beam entry

def convertToSequence(beam, classes):
	res = ""
	for l in beam:
		res += classes[l]
	return res

def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()

def ctcBeamSearch(mat, classes, lm, beamWidth, lmFactor, entropyThresh, lenContext, gt):
	"beam search as described by the paper of Hwang et al. and the paper of Graves et al."

	cache = {}

	blankIdx = len(classes)
	maxT, maxC = mat.shape

	# initialise beam state
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 1
	last.entries[labeling].prTotal = 1

	# go over all time-steps
	for t in range(maxT):
		if t < 30:
			continue
		curr = BeamState()

		# get beam-labelings of best beams
		bestLabelings = last.sort()[0][0:beamWidth]
		# bestLabelings = last.sort()[0:beamWidth]

		dummy_var = 0
		# Print out top 30 seqs (from bestLabelings) & their probs (from last)

		# go over best beams
		for labeling in bestLabelings:

			# probability of paths ending with a non-blank
			prNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				# we compute this before adding the beam to the state so we can
				# get the correct probability of the labeling (since a repeated
				# last char gets merged and therefore does not change the 
				# labeling, so we need to include the probability of a repeated
				# last char in the probability of the labeling)
				prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

			# probability of paths ending with a blank
			# this is computed for same logic as repeated last char
			prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

			# add beam at current time-step if needed
			addBeam(curr, labeling)

			# fill in data
			curr.entries[labeling].labeling = labeling
			curr.entries[labeling].prNonBlank += prNonBlank
			curr.entries[labeling].prBlank += prBlank
			curr.entries[labeling].prTotal += prBlank + prNonBlank
			curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
			curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling
			curr.entries[labeling].indices = last.entries[labeling].indices

			# extend current beam-labeling
			for c in range(maxC - 1):
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# keep track of matrix index that new char is located at
				# newIndices = deepcopy(curr.entries[labeling].indices)
				# newIndices.append(t)
				newIndices = curr.entries[labeling].indices + (t,)

				# if new labeling contains duplicate char at the end, only 
				# consider paths ending with a blank
				if labeling and labeling[-1] == c:
					# we can only extend a beam with the same char and it result
					# in a beam with duplicated char at the end if the previous
					# char was a blank (otherwise the repeated char would get
					# merged, as above), so we multiply by the blank probability
					prNonBlank = mat[t, c] * last.entries[labeling].prBlank
				else:
					# we can extend a beam with a different char regardless of
					# whether the previous char was a blank, so we multiply
					# by the total probability
					prNonBlank = mat[t, c] * last.entries[labeling].prTotal

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)
				
				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank += prNonBlank
				curr.entries[newLabeling].prTotal += prNonBlank
				curr.entries[newLabeling].indices = newIndices
				
				# apply LM
				tEntropy = entropy(mat[t])
				if tEntropy > entropyThresh:
					applyRNAModel(curr.entries[labeling],
								  curr.entries[newLabeling],
								  lm,
								  cache,
								  lmFactor,
								  lenContext)
 
		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	# sort by probability
	bestBeam = last.sort()

	# is the GT one of the possible beams??
	gt_tup = ()
	for c in gt:
		gt_tup += (classes.index(c),)
	
	candidates = bestBeam[0]
	for candidate in candidates:
		print(gt_tup == candidate)


	bestLabeling = bestBeam[0][0] # get most probable labeling
	# bestLabeling = last.sort()[0] # get most probable labeling

	# get indices corresponding to labeling
	bestLabelingIndices = bestBeam[1][0]

	# map labels to chars
	res = ''
	for l in bestLabeling:
		res += classes[l] # TODO: Improve efficiency

	return res, bestLabelingIndices
	# return res


def testBeamSearch():
	"test decoder"
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	print('Test beam search')
	expected = 'a'
	actual = ctcBeamSearch(mat, classes, None)
	print('Expected: "' + expected + '"')
	print('Actual: "' + actual + '"')
	print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
	testBeamSearch()
