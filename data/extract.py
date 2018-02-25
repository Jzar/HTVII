def extract(file):
	f = open(file,'r')
	tupleArray = []
	for line in f:
		zero = line[20:38]
		min = line.find('>')
		max = len(line)
		one = line[min+1:max].strip()
		tuple = [zero,one]
		tupleArray.append(tuple)
	return tupleArray
d = extract('test3.csv')
for i in range(len(d)):
	print d[i][0],d[i][1],'\n'
