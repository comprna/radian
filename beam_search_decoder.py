"""
This has been adapted from https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
"""


from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal = 0 # blank and non-blank
		self.prNonBlank = 0 # non-blank
		self.prBlank = 0 # blank
		self.prText = 1 # LM score
		self.lmApplied = False # flag if LM was already applied to this beam
		self.labeling = () # beam-labeling


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
		return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
	"calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
	if lm and not childBeam.lmApplied:
		c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
		c2 = classes[childBeam.labeling[-1]] # second char
		lmFactor = 0.01 # influence of language model
		bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
		childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
		childBeam.lmApplied = True # only apply LM once per beam entry


def applyRNAModel(parentBeam, childBeam, classes, lm, true_label, cache, lm_factor):
	"calculate RNA model score of child beam by taking score from parent beam and 6-mer probability of last six chars"
	if lm and not childBeam.lmApplied:
		if len(parentBeam.labeling) < 8:
			return

		# print(f"Ground truth: {true_label}")

		# print(f"Parent beam: {parentBeam.labeling}")
		# print(f"Last 8 in parent: {parentBeam.labeling[-8:]}")
		# print(f"Child: {childBeam.labeling}")

		context_tup = parentBeam.labeling[-8:]
		if context_tup not in cache:
			context = tf.one_hot(list(parentBeam.labeling[-8:]), 4).numpy()
			context = context.reshape(1,8,4)
			probs = lm.predict(context)
			cache[context_tup] = probs
		else:
			probs = cache[context_tup]

		# probs = lm.predict(context) ## INEFFICIENT: CACHE PREVIOUS PROBS???
		# print(f"LM probs: {probs}")

		new_char = childBeam.labeling[-1]
		prob_new_char = probs[0][new_char]
		# print(f"Prob new char: {prob_new_char}")

		# lmFactor = 0.1 # influence of language model
		prob_new_char = prob_new_char ** lm_factor # probability of seeing k-mer
		# print(f"Prob new char after factor {lm_factor}: {prob_new_char}\n\n")

		childBeam.prText = parentBeam.prText * prob_new_char # probability of whole sequence
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

def ctcBeamSearch(mat, classes, lm, true_label, beamWidth=6, lm_factor=0.1):
	"beam search as described by the paper of Hwang et al. and the paper of Graves et al."

	cache = {}

	blankIdx = len(classes)
	maxT, maxC = mat.shape

	# plt.imshow(np.transpose(mat), cmap="gray_r", aspect="auto")
	# plt.show()

	# initialise beam state
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 1
	last.entries[labeling].prTotal = 1

	# go over all time-steps
	for t in range(maxT):
		curr = BeamState()

		# get beam-labelings of best beams
		bestLabelings = last.sort()[0:beamWidth]

		# go over best beams
		for labeling in bestLabelings:

			# probability of paths ending with a non-blank
			prNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

			# probability of paths ending with a blank
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

			# extend current beam-labeling
			for c in range(maxC - 1):
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
				if labeling and labeling[-1] == c:
					prNonBlank = mat[t, c] * last.entries[labeling].prBlank
				else:
					prNonBlank = mat[t, c] * last.entries[labeling].prTotal

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)
				
				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank += prNonBlank
				curr.entries[newLabeling].prTotal += prNonBlank
				
				# apply LM
				# print("applying model!")
				applyRNAModel(curr.entries[labeling], curr.entries[newLabeling], classes, lm, true_label, cache, lm_factor)
 
		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	 # sort by probability
	bestLabeling = last.sort()[0] # get most probable labeling

	# map labels to chars
	res = ''
	for l in bestLabeling:
		res += classes[l]

	return res


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
