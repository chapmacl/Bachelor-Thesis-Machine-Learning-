'''
Created on May 2, 2018

@author: chapmacl
'''

import csv
import re

r = csv.reader(open('flu_tweets.csv'))
lines = list(r)
text = lines[0][0]
for x in range(0, lines.__len__()):
    text = lines[x][1]
    text = re.sub(r"https\S+", "", text)
    text = re.sub(r"@\S+", "", text) 
    text = re.sub("RT", "", text)
    text = re.sub("\n", " ", text)
    lines[x][1] = text

writer = csv.writer(open('test.csv', "a"))
writer.writerows(lines)

