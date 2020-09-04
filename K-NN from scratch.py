import csv
import random
import math
import operator

with open('iris.data.txt', 'r') as csvfile:

 lines = csv.reader(csvfile)
 for row in lines :
  print (', '.join(row))

def loadDataset(filename, split, trainingSet=[] , testSet=[]):

 with open(filename, 'r') as csvfile:

    lines = csv.reader(csvfile)

    dataset = list(lines)

    for x in range(len(dataset)-1):

        for y in range(4):

            dataset[x][y] = float(dataset[x][y])

        if random.random() < split:

            trainingSet.append(dataset[x])

        else:

            testSet.append(dataset[x])
trainingSet=[]
testSet=[]

loadDataset('iris.data.txt', 0.66, trainingSet, testSet)
print('Train: ' + repr(len(trainingSet))) # 105
print ('Test: ' + repr(len(testSet))) # 44

#****************************************************Euclidian distance **********************************************************


def euclideanDistance(instance1, instance2, length):
 distance = 0
 for x in range(length):
  distance += pow((instance1[x] - instance2[x]), 2)
 return math.sqrt(distance)

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print ('Distance: ' + repr(distance)) #3.46

def MinhatanDistance(instance1,instance2,length):
    sum = 0

    for i in range(length):
        for j in range(i + 1, length):
            sum += (abs(instance1[i] - instance1[j]) +
                    abs(instance1[i] - instance2[j]))

    return sum

def getNeighbors(trainingSet, testInstance, k):

 distances = []

 length = len(testInstance)-1

 for x in range(len(trainingSet)):

  dist = MinhatanDistance(testInstance, trainingSet[x], length)

     #dist = euclideanDistance(testInstance, trainingSet[x], length)

  distances.append((trainingSet[x], dist))

  distances.sort(key=operator.itemgetter(1))

  neighbors = []

  for x in range(k):

   neighbors.append(distances[x][0])

   return neighbors

trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)

def getResponse(neighbors):

 classVotes = {}

 for x in range(len(neighbors)):

  response = neighbors[x][-1]

  if response in classVotes:

   classVotes[response] += 1

  else:

   classVotes[response] = 1

   sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

 return sortedVotes[0][0]
neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbors)
print(response)

#******************************************************************************************************************************

def getAccuracy(testSet, predictions):
 correct = 0
 for x in range(len(testSet)):
  if testSet[x][-1] is predictions[x]:
   correct += 1

 return (correct/float(len(testSet))) * 100.0


testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)

#************************************************** Main **********************************************************************

def main():

 trainingSet=[]

 testSet=[]

 split = 0.67

 loadDataset('iris.data.txt', split, trainingSet, testSet)

 print ('Train set: ' + repr(len(trainingSet)))

 print ('Test set: ' + repr(len(testSet)))

 predictions=[]

 k = 3

 for x in range(len(testSet)):

  neighbors = getNeighbors(trainingSet, testSet[x], k)

  result = getResponse(neighbors)

  predictions.append(result)

  print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

accuracy = getAccuracy(testSet, predictions)

print('Accuracy: ' + repr(accuracy) + '%')

main()

