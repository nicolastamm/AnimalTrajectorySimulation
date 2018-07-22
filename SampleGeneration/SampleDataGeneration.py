#HC SVNT DRACONES , VADE RETRO
#@author: Nicolas Andreas Tamm Garetto

from SampleGeneration.perlinnoise import *
import matplotlib.pyplot as plt
import numpy as np

#The general idea is to simulate how a bird would walk around a 15x8mts area.
#For this we create a field which simulates how "interesing" and "uninteresting" things are distributed in this area.
#Then we perform gradient descent with momentum, while using the x,y position on the interests-field as a loss function.

#Initially how interesting things are placed in the field is static.(For now)
#This has 2 components (For now): A random one, implemented with Perlin Noise,
#and a definite one: the natural aversion of the bird to crash against the nets.
def createInterestFieldSimulation(randomSeed):
#Perlin noise Component
    #setting up the parameters for the Perlin noise.
    lin01 = np.linspace(0, 3, 80, endpoint=False)
    lin02 = np.linspace(0, 3, 150)
    x, y = np.meshgrid(lin01, lin02)
    perlinComponent = perlin(x , y , seed=randomSeed) #See -> perlinnoise.py
    # Normalizing Perlin Noise
    minVal = np.amin(perlinComponent)
    maxVal = np.amax(perlinComponent)
    for  i in range(len(perlinComponent)):
        for j in range(len(perlinComponent[0])):
            #(*) linear space from 0 to 0.Y
           perlinComponent[i][j] = np.interp(perlinComponent[i][j], [minVal , maxVal] , [0.0 , 0.5])

#Proximity to border component
    proximityComponent = np.zeros(perlinComponent.shape , dtype='float')
    #Descending order of these results in the borders getting the higher values.
    #Otherwise, the center would have the highest value.
    horLinSpace = np.linspace(2.0, 0.0 , 75)
    vertLinSpace = np.linspace(2.0 , 0.0 , 40)

    #Borders have e^1 as a value and the middle has the lowest value.
    #We enforce symmetry along the middle of the axes.
    #the max(something clever , e^1/2) is for tuning down the importance of corners
    for i in range(len(proximityComponent)):
        for j in range(len(proximityComponent[0])):
            if i > 75: # this ...
                proximityComponent[i][j] += max(np.exp(horLinSpace[75 - i]) , np.exp(1.0) / 0.5)
            else:
                proximityComponent[i][j] += max(np.exp(horLinSpace[i-1]) , np.exp(1.0) / 0.5)
            if j > 40: #and this is how we do symmetry. Rough, but it works.
                proximityComponent[i][j] += max(np.exp(vertLinSpace[40 - j]) , np.exp(1.0) / 0.5)
            else:
                proximityComponent[i][j] += max(np.exp(vertLinSpace[j-1]) , np.exp(1.0) / 0.5)

    # Normalizing Distance Factor
    minVal = np.amin(proximityComponent)
    maxVal = np.amax(proximityComponent)
    for  i in range(len(proximityComponent)):
        for j in range(len(proximityComponent[0])):
            #(**)linear space from 0 to 0.X
           proximityComponent[i][j] = np.interp(proximityComponent[i][j], [minVal , maxVal] , [0.0 , 0.5])
    #(*) + (**) gives values in the [0.0 , 1.0] range, since X + Y = 1
    #Controling X and Y modifies how important each component is for the final "loss function"
    #For the gradient descent, we only really care about the... gradients.
    return  proximityComponent + perlinComponent
    #proximityComponent +

def gradientDescent(position , landscapeGradients , momentum):
    eta = 20.0 #learning rate
    alpha = 0.9 #momentum retention factor
    horStepSize = eta * landscapeGradients[0][int(position[0])][int(position[1])] + alpha * momentum[0]
    vertStepSize = eta * landscapeGradients[1][int(position[0])][int(position[1])] + alpha * momentum[1]
    momentum = [horStepSize , vertStepSize]
    return momentum
def main(frames):
    j = 0
    for x in range(1000):
        interestsField = createInterestFieldSimulation(j+1000)
        interestsGradient = np.gradient(interestsField)
        momentum = [0.0 , 0.0]
        path = np.zeros(shape=[frames + 1,3])
        i = 0
        xstartpos = np.random.uniform(0.0, 150.0, 1)[0]
        ystartpos = np.random.uniform(0.0, 80.0, 1)[0]
        position = [xstartpos , ystartpos]
        print(position)
        while i < frames:
            j+=1
            i+=1
            gradDescStep = gradientDescent(position , interestsGradient , momentum)
            #0.33 is the average flight velocity of a Greylag Goose per Frame, iff 1s ~ 60 Frames
            if(euclDist([position[0] - gradDescStep[0] , position[1] - gradDescStep[1]] , path[i-1]) > 0.33):
                gradDescStep[0] /= 2.0
                gradDescStep[1] /= 2.0
            position[0] -= gradDescStep[0]
            position[1] -= gradDescStep[1]
            momentum = gradDescStep
            if euclDist([position[0] , position[1]] , path[i-1]) < 0.11:
                interestsGradient = np.gradient(createInterestFieldSimulation(j + 1000))
                i-=1
            path[i] = [position[0], position[1] , i]
            print("{}% , {}".format(round((i / frames) * 100,2) , i))
        plt.imshow(interestsField)
        plt.scatter(path[: ,1] , path[: , 0], c="r" , s=0.5)
       # plt.show()
        plt.savefig("synthBirdFlightImFluidSlower{}.png".format(x+1))
        f = open("synthBirdFlightFluidSlower{}.txt".format(x+1),"w+")
        plt.gcf().clear()
        for i in range(len(path)):
            f.write("{},{},{}\n".format(i,path[i][0],path[i][1]))

def euclDist(x , y):
    return np.sqrt((x[0] - y[0]) **2 + (x[1] - y[1]) **2)

main(12700)