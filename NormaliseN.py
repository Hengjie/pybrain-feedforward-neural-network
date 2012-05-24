## George Coghill 13 April 2011
#This is intended to be a general purpose program to normalise data sets
#It will find the highest absolute value within the input attributes and normalise
#all the attribute values through division by this number
from math import *
Input = 4
Output = 2
InputAttributeData = []
OutputClassData = []
InFile = 'Blood1.dat'
OutFile = 'Blood1N.dat'
maxValue = 0.0
lineNo = 1
#Configure input file (text) into input list and output list
print("FILE is %s" % InFile)
inFile = open(InFile, 'r')
outFile = open(OutFile, 'w')
for line in inFile.readlines():
    L = line.split(" ")    #L should be a list of input attributes and classes on line
    inSample = []          #re-initialise for each incoming line
    outSample = []
    for i in range(Input):
        sample = float(L[i])
        inSample.append(sample)
        if  fabs(sample) > fabs(maxValue) :
            maxValue = fabs(sample)
    for j in range(Output):
        outSample.append(float(L[j+Input]))
    InputAttributeData.append(inSample)
    OutputClassData.append(outSample)
for i in range(len(InputAttributeData)):
    for j in range(Input):
        InputAttributeData[i][j] /= maxValue
        outFile.write("%f" % InputAttributeData[i][j],)
        outFile.write(' ',)
    for j in range(Output):
        outFile.write("%f" % OutputClassData[i][j],)
        if j != Output - 1 :
           outFile.write(' ',)
        else:
           outFile.write('\n')
    print lineNo
    lineNo += 1
outFile.close()
inFile.close()
           
        
        
