import matplotlib.pyplot as plt
import numpy as np
import logging
import testcarte
logging.basicConfig(level=logging.INFO, format=' %(levelname)s - %(message)s')

def contour(carte):
    '''Trouve les contours dans une image dont les pixels valent soit 0 soit 1.
    carte: ndarray'''
    rep = np.zeros(np.shape(carte),np.int8)
    for i in range(np.shape(carte)[0]):
        for j in range(np.shape(carte)[1]):
            if carte[i,j] == 1:
                if i == 0 and j == 0:
                    if np.min(carte[:2,:2]) == 0:
                        rep[0,0] = 1
                elif i == np.shape(carte)[0]-1 and j == 0:
                    if np.min(carte[-2:,:2]) == 0:
                        rep[-1,0] = 1
                elif i == 0 and j == np.shape(carte)[1]-1:
                    if np.min(carte[:2,-2:]) == 0:
                        rep[0,-1] = 1
                elif i == np.shape(carte)[0]-1 and j == np.shape(carte)[1]-1:
                    if np.min(carte[-2:,-2:]) == 0:
                        rep[-1,-1] = 1
                elif i == 0:
                    if np.min(carte[:2,j-1:j+2]) == 0:
                        rep[0,j] = 1
                elif i == np.shape(carte)[0]-1:
                    if np.min(carte[-2:,j-1:j+2]) == 0:
                        rep[-1,j] = 1
                elif j == 0:
                    if np.min(carte[i-1:i+2,:2]) == 0:
                        rep[i,0] = 1
                elif j == np.shape(carte)[1]-1:
                    if np.min(carte[i-1:i+2,-2:]) == 0:
                        rep[i,-1] = 1
                else:
                    if np.min(carte[i-1:i+2,j-1:j+2]) == 0:
                        rep[i,j] = 1
    return rep

terres = testcarte.carte_4(testcarte.carte_3(100,10,0.35),1)
villages = testcarte.carte_3(500,1000,0.993,0.9,np.array(terres,int)+20*np.array(contour(terres),int),terres)
montagnes = terres * testcarte.carte_4(testcarte.carte_3(100,100,0.95,0.5),1)
arbres = terres * (1 - villages) * testcarte.carte_4(testcarte.carte_3(100,500,0.65,0.1),1)
cartecouleur = np.zeros((500,500,3),np.uint8)

for i in range(500):
    for j in range(500):
        if terres[i,j] == 0:
            cartecouleur[i,j] = np.array([0,0,153])
        elif villages[i,j] == 1:
            cartecouleur[i,j] = np.array([153,64,0])
        elif montagnes[i,j] == 1 and arbres[i,j] == 1:
            cartecouleur[i,j] = np.array([48,65,48])
        elif arbres[i,j] == 1:
            cartecouleur[i,j] = np.array([0,65,0])
        elif montagnes[i,j] == 1:
            cartecouleur[i,j] = np.array([77,77,77])
        else:
            cartecouleur[i,j] = np.array([0,255,0])

plt.imshow(cartecouleur)
plt.show()