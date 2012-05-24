## George Coghill 13 April 2011
#This is intended to be a general purpose program to normalise data sets
#It will find the highest absolute value within each input attribute and normalise
#on a per attribute basis. It will also ensure a 
#zero average value per attribute
#Input/output parameter values
from math import *
Input = 4
Output = 2
InputAttributeData = []
OutputClassData = []
InFile = 'Blood1.dat'
OutFile = 'BloodZMC.dat'
maxValue = []
sample = []
total = []
average = []
lineNo = 1
#Configure input file (text) into input list and output list
print("FILE is %s" % InFile)
inFile = open(InFile, 'r')
outFile = open(OutFile, 'w')
#Initialise vectors
for i in range(Input):
    maxValue.append(0.0)
    sample.append(0.0)
    total.append(0.0)
    average.append(0.0)

for line in inFile.readlines():
    L = line.split(" ")    #L should be a list of input attributes and classes on line
    inSample = []          #re-initialise for each incoming line
    outSample = []
    
    for i in range(Input):
        sample[i] = float(L[i])
        total[i] += sample[i]   #Find the total of each input
        inSample.append(sample[i])  #build list of samples for attribute
        if  fabs(sample[i]) > fabs(maxValue[i]) :
           maxValue[i] = sample[i]
    for j in range(Output):
        outSample.append(float(L[j+Input]))    #build output columns
    InputAttributeData.append(inSample)
    OutputClassData.append(outSample)
for i in range(len(InputAttributeData)):
    for j in range(Input):
        average[j] = total[j]/len(InputAttributeData)    #find each average
        InputAttributeData[i][j] -= average[j]
        InputAttributeData[i][j] /= maxValue[j]
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
           
        
        
