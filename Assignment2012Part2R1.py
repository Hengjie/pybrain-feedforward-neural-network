#First draft G Coghill on April 6 2011
#The PyBrain BackProp Simulator with GUI
#Modified April 25 2012, now revision 1

#result file
result = 'result.dat'

#Import plotter
from matplotlib import pylab
from Tkinter import *	#2011
import sys
import os.path
import string

from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer, BiasUnit, TanhLayer
from pybrain.structure.networks import FeedForwardNetwork, Network
from pybrain.structure.connections import FullConnection

class BrainApp:
    TkField = []
    #Parameters (attributes) and values
    parameters = {'INPUT': '4',     #No of input dimensions
                  'OUTPUT': '3',    #No of Output Class
                  'HIDDEN0': '3',   #No of Hidden Neurons 1st layer
                  'HIDDEN1': '2',   #Second Layer
                  'HIDDEN_S/T': 'T', #Hidden layer activations (S)oftMax or (T)anh
                  'OUTPUT_S/L': 'S', #Output layer activation (S)oftMax or (L)inear
                  'LEARNING_RATE': '0.05',
                  'MOMENTUM': '0.0',
                  'BIAS': 'True',
                  'EPOCHS': '120',  #No of training cycles used
                  'WEIGHT_DECAY': '0.0',
                  'SPLIT_DATA': '0.1',
                  'UPPER_LIMIT': '0.6',  #Higher than this, taken as 1.0
                  'LOWER_LIMIT': '0.4' } #Less than this, taken as zero

    #Configure input file (text) into input list and output list
    def readMyData(self, filename):
       print("FILE is %s" % filename)
       file = open(filename, 'r')
       for line in file.readlines():
           L = line.split(" ")
           inSample = []
           outSample = []
           for i in range(int(self.parameters['INPUT'])):
              inSample.append(float(L[i]))
           for j in range(int(self.parameters['OUTPUT'])):
              outSample.append(float(L[j+int(self.parameters['INPUT'])]))
          
           self.myDataset.addSample(inSample,outSample)

    #current Error Measure - You could add your own
    def SumSquareError(self, Actual, Desired):
       error = 0.
       for i in range(len(Desired)):
          for j in range(len(Desired[i])):
             error = error + ((Actual[i])[j] - (Desired[i])[j])*((Actual[i])[j] - (Desired[i])[j])
       return error

    def RunSimulator(self):
        self.myDataset = ClassificationDataSet(int(self.parameters['INPUT']), int(self.parameters['OUTPUT']))
        #Load float format iris dataset
        self.readMyData(filename)
        #create baseline network
        self.network = FeedForwardNetwork()
        #Selection of hidden layers (1 or 2)
        if int(self.parameters['HIDDEN1']) == 0 :
            #Build Architecture (One hidden layer)
            inLayer = LinearLayer(int(self.parameters['INPUT']))
            if self.parameters['HIDDEN_S/T'] == 'T':
                hiddenLayer0 = TanhLayer(int(self.parameters['HIDDEN0']))
            else:
                hiddenLayer0 = SoftmaxLayer(int(self.parameters['HIDDEN0']))
            if self.parameters['OUTPUT_S/L'] == 'S': 
                outLayer = SoftmaxLayer(int(self.parameters['OUTPUT']))
            else:
                outLayer = LinearLayer(int(self.parameters['OUTPUT']))
            self.network.addInputModule(inLayer)
            self.network.addModule(hiddenLayer0)
            self.network.addOutputModule(outLayer)
 
            #Make connections
            in_to_hidden = FullConnection(inLayer, hiddenLayer0)
            hidden_to_out = FullConnection(hiddenLayer0, outLayer)
            self.network.addConnection(in_to_hidden)
            self.network.addConnection(hidden_to_out)
            if self.parameters['BIAS'] == 'True':
                bias = BiasUnit('bias')
                self.network.addModule(bias)
                bias_to_hidden0 = FullConnection(bias, hiddenLayer0)
                bias_to_out = FullConnection(bias, outLayer)            
                self.network.addConnection(bias_to_hidden0)
                self.network.addConnection(bias_to_out)  
        elif int(self.parameters['HIDDEN0']) == 0 :
            print "Cannot delete layer 0"
            sys.exit()
        else:   #Two hidden layers
            #Build Architecture
            inLayer = LinearLayer(int(self.parameters['INPUT']))
            if self.parameters['HIDDEN_S/T'] == 'T':
                hiddenLayer0 = TanhLayer(int(self.parameters['HIDDEN0']))
                hiddenLayer1 = TanhLayer(int(self.parameters['HIDDEN1']))
            else:
                hiddenLayer0 = SoftmaxLayer(int(self.parameters['HIDDEN0']))
                hiddenLayer1 = SoftmaxLayer(int(self.parameters['HIDDEN1']))
            if self.parameters['OUTPUT_S/L'] == 'S': 
                outLayer = SoftmaxLayer(int(self.parameters['OUTPUT']))
            else:
                outLayer = LinearLayer(int(self.parameters['OUTPUT']))
            self.network.addInputModule(inLayer)
            self.network.addModule(hiddenLayer0)
            self.network.addModule(hiddenLayer1)
            self.network.addOutputModule(outLayer)
            
            #Make connections
            in_to_hidden = FullConnection(inLayer, hiddenLayer0)
            hidden_to_hidden = FullConnection(hiddenLayer0, hiddenLayer1)
            hidden_to_out = FullConnection(hiddenLayer1, outLayer)
            self.network.addConnection(in_to_hidden)
            self.network.addConnection(hidden_to_hidden)
            self.network.addConnection(hidden_to_out)
            if self.parameters['BIAS'] == 'True':
                bias = BiasUnit('bias')
                self.network.addModule(bias)
                bias_to_hidden0 = FullConnection(bias, hiddenLayer0)
                bias_to_hidden1 = FullConnection(bias, hiddenLayer1)
                bias_to_out = FullConnection(bias, outLayer)            
                self.network.addConnection(bias_to_hidden0)
                self.network.addConnection(bias_to_hidden1)
                self.network.addConnection(bias_to_out)  

        # topologically sort the units in network
        self.network.sortModules()

        # split the data randomly into 90% training, 10% test = 0.1
        testData, trainData = self.myDataset.splitWithProportion(float(self.parameters['SPLIT_DATA']))

        #create the trainer environment for backprop and  train network
        trainer = BackpropTrainer(self.network, dataset = trainData, learningrate = \
        float(self.parameters['LEARNING_RATE']), momentum=float(self.parameters['MOMENTUM']), \
        weightdecay=float(self.parameters['WEIGHT_DECAY']))

        #Data workspace
        TrainingPoints = []
        TestPoints = []
        xAxis = []
        #Run for specified number of Epochs
        for i in range(int(self.parameters['EPOCHS'])):
            trainer.trainEpochs(1)
            trnresult = self.SumSquareError(self.network.activateOnDataset(dataset=trainData), trainData['target'])
            tstresult = self.SumSquareError(self.network.activateOnDataset(dataset=testData), testData['target'])
            #Print Current Errors (comment out when not needed)
            print "epoch: %4d" % trainer.totalepochs, \
            " train error: %5.2f" % trnresult, \
            " test error: %5.2f" % tstresult
            #Build Lists for plotting 
            TrainingPoints.append(trnresult)
            TestPoints.append(tstresult)
            xAxis.append(i)

        #Output Results
        self.AnalyzeOutput(testData)
        pylab.plot(xAxis, TrainingPoints, 'b-')
        pylab.plot(xAxis,TestPoints, 'r-')
        pylab.xlabel('EPOCHS')
        pylab.ylabel('Sum Squared Error')
        pylab.title('Plot of Training Errors')
        pylab.show()

    #Create Parameters GUI
    def __init__(self, parent):
        self.myparent = parent
        r = 0
        for c in self.parameters.keys():  
            Label(parent, text=c, relief=RIDGE,  width=25).grid(row=r, column=0)
            self.TkField.append(Entry(parent))
            self.TkField[r].grid(row=r, column=1)
            self.TkField[r].insert(0,"                %s" % self.parameters[c])
            self.TkField[r].bind("<Button-1>", self.ClearWidget)
            self.TkField[r].bind("<Return>", self.ok)
            r += 1
        Button(parent,text='RUN', command=self.RunSimulator).grid(row=r, column=0)
        Button(parent,text='END', command=self.terminate).grid(row=r, column=1)

    #Enter new data (called when <Return> key is pressed after data entry)
    def ok(self, event):
        position = int(event.widget.grid_info()['row'])  #get the parameter position of the <Return> event
        self.TkField[position].insert(0,'                ')
        self.parameters[self.parameters.keys()[position]] = self.TkField[position].get() #get new data entry
        self.TkField[position].delete(0,END)
        self.TkField[position].insert(0,self.parameters[self.parameters.keys()[position]])
        self.TkField[position].icursor(0)

    #END button is pressed
    def terminate(self):
        self.myparent.destroy()

    #Clear Data Field in GUI (lefthand mouse key is pressed)
    def ClearWidget(self, event):
        position = int(event.widget.grid_info()['row'])  #get the parameter position of the <Button-1> event  
        self.TkField[position].delete(0,END)

    #Analyse the Test Data Set
    #Save the actual and desired results for tes data in 'result.dat'
    #Print the Correct, Wrong and Unknown Statistics
    def AnalyzeOutput(self, testData):
        #Compare actual test results (writes to data file 'result.dat')
        total = { 'good' : 0,
                  'bad' : 0,
                  'unknown' : 0 }        
        actualTestOutput = self.network.activateOnDataset(dataset=testData)
        desiredTestOutput = testData['target']
        resultFile = open(result, 'w')
        resultFile.write( "   Test Outputs\n")
        resultFile.write(" Actual    Desired")
        for m in range(len(actualTestOutput)):        #say, 15 for iris data (10%)
            resultFile.write('\n')
            sample = { 'good': 0,
                       'bad' : 0,
                       'unknown' : 0 }
            for n in range(int(self.parameters['OUTPUT'])):   #say, 3 classes as in iris data
               actual = float(actualTestOutput[m][n])   #class by class
               resultFile.write( "  %2.1f" % actual,)
               resultFile.write( '\t',)
               desired = float(desiredTestOutput[m][n])
               resultFile.write( "    %2.1f\n" % desired )
              
               #for classifier, calculate the errors
               upper = float(self.parameters['UPPER_LIMIT'])
               lower = float(self.parameters['LOWER_LIMIT'])

               #Check for silly values
               if upper >= 1.0 or lower >= 1.0 or lower > upper :
                   print "Illegal setting of upper or lower decision limit"
                   sys.exit()
               #My logic to built result determined by the values of upper and lower limits
               if desired > upper and actual > upper :
                   sample['good'] += 1
               elif desired < lower and actual < lower :
                   sample['good'] += 1
               elif desired > upper and actual < lower :    #corrected 25th April
                   sample['bad'] += 1
               elif desired < lower and actual > upper :    #corrected 25th April
                   sample['bad'] += 1
               else :
                   sample['unknown'] += 1
            if sample['good'] == int(self.parameters['OUTPUT']) :
                total['good'] += 1
            elif sample['bad'] > 0 :
                total['bad'] += 1
            elif sample['bad'] == 0 and sample['unknown'] > 0 :
                total['unknown'] += 1

        #Print Test Result Analysis to Screen
        print  #New line
        percentage = 100.0*float(total['good'])/float(len(actualTestOutput))
        print " Correct\t= %f" % percentage
        percentage = 100.0*float(total['bad'])/float(len(actualTestOutput))
        print " Wrong\t\t= %f" % percentage
        percentage = 100.0*float(total['unknown'])/float(len(actualTestOutput))
        print " Unknown\t= %f" % percentage
        resultFile.close()

###main Program
filename = sys.argv[1] 
root = Tk()
w = Label(root, text="CMPSYS406 Assignment 2")
app = BrainApp(root)
root.mainloop()
