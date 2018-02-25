import time
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
f = open( "processedTweets",'a')
for j in range(len(d)):
	d[j][0] = d[j][0].split(" ")
	time.sleep(0.25) #We are limited in our api calls per seconds
	f.write( str(d[j][0][0])+","+str(d[j][0][1])+","+str(sent(d[j][1]))+"\n")
