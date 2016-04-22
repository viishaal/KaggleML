import time

_SUBMISSIONS_TO_VOTE_ = ["adaboost_1461223085.csv", "etc_1461280206.csv", "etc_1461332862.csv"]
_OUTPUT_FILE_ = 'Submissions/voted_{}.csv'

if __name__ == "__main__":
	f = open(_OUTPUT_FILE_.format(int(time.time())), "w")
	f.write("Id,Prediction\n")
	flines = []
	for s in _SUBMISSIONS_TO_VOTE_:
		flines.append(open("Submissions/"+s, "r").readlines())

	for i in range(1,len(flines[0])):
		result = 0
		for j in flines:
			vote = int(j[i].split(',')[1])
			result = result + vote
			print result, vote
		if result > 0:
			result = 1
		else:
			result = -1
		f.write(str(i)+","+str(result)+"\n")
	f.close()
