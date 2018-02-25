from sentiment import sent
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
d = extract('twitterData.txt')
f = open( "processedTweets",'w')
for i in range(len(d)):
	d[i][0] = d[i][0].split(" ")
	print (d[i][0][1]) 
	f.write( str(d[i][0][0])+","+str(d[i][0][1])+","+str(sent(d[i][1]))+"\n")
