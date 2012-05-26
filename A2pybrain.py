#First draft G Coghill on April 6 2011
#The PyBrain BackProp Simulator with GUI
#Modified April 25 2012, now revision 1
#Modified by Hengjie to make it command line only

#result file
result = 'result.dat'

#Import plotter
#from matplotlib import pylab
import sys
import os.path
import string

from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LinearLayer, SigmoidLayer, SoftmaxLayer, BiasUnit, TanhLayer
from pybrain.structure.networks import FeedForwardNetwork, Network
from pybrain.structure.connections import FullConnection

from multiprocessing import Process, Queue, Manager
from pprint import pprint 

class BrainApp(Process):

    #Parameters (attributes) and values
    parameters = {'INPUT': '4',     #No of input dimensions
                  'OUTPUT': '2',    #No of Output Class
                  'HIDDEN0': '9',   #No of Hidden Neurons 1st layer
                  'HIDDEN1': '9',   #Second Layer
                  'HIDDEN_S/T': 'T', #Hidden layer activations (S)oftMax or (T)anh
                  'OUTPUT_S/L': 'S', #Output layer activation (S)oftMax or (L)inear
                  'LEARNING_RATE': '0.15',
                  'MOMENTUM': '0.0',
                  'BIAS': 'True',
                  'EPOCHS': '25',  #No of training cycles used
                  'WEIGHT_DECAY': '0.0',
                  'SPLIT_DATA': '0.1',
                  'UPPER_LIMIT': '0.6',  #Higher than this, taken as 1.0
                  'LOWER_LIMIT': '0.4' } #Less than this, taken as zero

    def __init__(self, output, filename, epoch=120, learningrate=0.15, hidden=9):
      self.filename = filename

      self.parameters['EPOCHS'] = epoch
      self.parameters['LEARNING_RATE'] = learningrate
      self.parameters['HIDDEN0'] = hidden
      self.parameters['HIDDEN1'] = hidden
      self.output = output
      super(BrainApp, self).__init__()

    def run(self):
      self.output.append(self.RunSimulator());

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
        #pprint(self.parameters)

        self.myDataset = ClassificationDataSet(int(self.parameters['INPUT']), int(self.parameters['OUTPUT']))
        #Load float format iris dataset
        self.readMyData(self.filename)
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
            #print "epoch: %4d" % trainer.totalepochs, \
            #" train error: %5.2f" % trnresult, \
            #" test error: %5.2f" % tstresult
            #Build Lists for plotting 
            TrainingPoints.append(trnresult)
            TestPoints.append(tstresult)
            xAxis.append(i)

        #Output Results
        return self.AnalyzeOutput(testData)

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

        total_ret = {}

        #Print Test Result Analysis to Screen
        print  #New line
        percentage = 100.0*float(total['good'])/float(len(actualTestOutput))
        total_ret['correct_percentage'] = percentage
        print " Correct\t= %f" % percentage
        percentage = 100.0*float(total['bad'])/float(len(actualTestOutput))
        total_ret['bad_percentage'] = percentage
        print " Wrong\t\t= %f" % percentage
        percentage = 100.0*float(total['unknown'])/float(len(actualTestOutput))
        total_ret['unknown_percentage'] = percentage
        print " Unknown\t= %f" % percentage
        resultFile.close()

        total_ret['epoch'] = self.parameters['EPOCHS']
        total_ret['leanringrate'] = self.parameters['LEARNING_RATE']
        total_ret['hidden'] = self.parameters['HIDDEN0']

        return total_ret

# thanks http://stackoverflow.com/questions/7267226/range-for-floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

###main Program
filename = sys.argv[1]
overall_results = Manager().list()

for epoch in range(25, 125, 20):
  print "====================="
  print "epoch: {0}".format(epoch)  

  # different learning rates
  for learningrate in frange(0.00, 1, 0.1):

    workers = []

    print "learning: {0}".format(learningrate)
    # run it three times
    for x in range(6):
      brain_thread = BrainApp(overall_results, filename, epoch, learningrate, 9)
      brain_thread = BrainApp(overall_results, filename, epoch, learningrate, 18)
      workers.append(brain_thread)

      brain_thread.start()

    for brain_thread in workers:
        brain_thread.join()

# analyse the data
print "epoch\tlearnRate\rhidden\tcorrect\tbad\tunknown\n"
for row in overall_results:
  print "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(row['epoch'], row['leanringrate'], row['hidden'], round(row['correct_percentage'], 2), round(row['bad_percentage'], 2), round(row['unknown_percentage'], 2))
