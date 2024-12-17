## Définitions
import numpy as np
import logging

##def carte_1(dim,n,sea=0.5,ampl=10,degrade=False):
##    '''dim: Côté de la carte en pixels. Nombre entier strictement positif. Valeur recommandée de quelques centaines.
##    n: Nombre de champs. Nombre entier strictement positif.
##    sea: Proportion d\'eau. Nombre compris entre 0 et 1
##    ampl: Amplitude des champs. Nombre strictement positif.
##    degrade: Vrai pour laisser la trace des champs. Booléen'''
##    assert dim > 0 and n > 0 and n <= dim**2 and sea >= 0 and sea <= 1 and ampl > 0, 'Les valeurs ne sont pas dans les bons intervalles.'
##    sources = np.zeros((n,3))
##    sources[:,:2] = np.random.randint(0,dim,(n,2))
##    sources[:,2:] = np.random.uniform(-ampl*sea*2,ampl*(1-sea)*2,(n,1))
##    carte = np.zeros((dim,dim))
##    for i in range(dim):
##        for j in range(dim):
##            ghjk = 0
##            for k in range(n):
##                if sources[k,0] == i and sources[k,1] == j:
##                    ghjk += sources[k,2] * 3
##                else:
##                    ghjk += sources[k,2]/((sources[k,0]-i)**2+(sources[k,1]-j)**2)
##            if degrade:
##                ghjk = (ghjk+1)/2
##                if ghjk > 1:
##                    ghjk = 1
##                elif ghjk < 0:
##                    ghjk = 0
##            else:
##                if ghjk < 0:
##                    ghjk = 0
##                else:
##                    ghjk = 1
##            carte[i,j] = ghjk
##    return carte
##
##def carte_2(dim,n,sea=0.5,alea=0.642):
##    '''dim: Côté de la carte en pixels. Nombre entier strictement positif. Valeur recommandée de quelques centaines.
##    n: Nombre de sources. Nombre entier strictement positif.
##    sea: Proportion d\'eau. Nombre compris entre 0 et 1
##    alea: Amplitude des variations aléatoires. Nombre positif ou nul. A peu d\'effet marginal au-delà de 1'''
##    assert dim > 0 and n > 0 and n <= dim**2 and sea >= 0 and sea <= 1 and alea >= 0, 'Les valeurs ne sont pas dans les bons intervalles.'
##    sources = np.zeros((n,3))
##    sources[:,:2] = np.random.randint(0,dim,(n,2))
##    sources[:,2:] = 2*np.random.binomial(1,1-sea,(n,1))-1
##    carte = np.zeros((dim,dim))
##    wcarte = np.zeros((dim,dim))
##    for i in range(n):
##        wcarte[int(sources[i,0]),int(sources[i,1])] = sources[i,2]
##    while 0 in carte:
##        for i in range(dim):
##            for j in range(dim):
##                if i == 0 or i == dim-1 or j == 0 or j == dim-1:
##                    if i == 0 and j == 0:
##                        if (carte[:2,:2] != np.zeros((2,2))).any() and carte[0,0] == 0:
##                            ghjk = np.average(carte[:2,:2],None,[[1,1],[1,0.7]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif i == dim-1 and j == 0:
##                        if (carte[dim-2:,:2] != np.zeros((2,2))).any() and carte[dim-1,0] == 0:
##                            ghjk = np.average(carte[dim-2:,:2],None,[[1,0.7],[1,1]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif i == 0 and j == dim-1:
##                        if (carte[:2,dim-2:] != np.zeros((2,2))).any():
##                            ghjk = np.average(carte[:2,dim-2:],None,[[1,1],[0.7,1]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif i == dim-1 and j == dim-1:
##                        if (carte[dim-2:,dim-2:] != np.zeros((2,2))).any():
##                            ghjk = np.average(carte[dim-2:,dim-2:],None,[[0.7,1],[1,1]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif i == 0:
##                        if (carte[:2,j-1:j+2] != np.zeros((2,3))).any():
##                            ghjk = np.average(carte[:2,j-1:j+2],None,[[1,1,1],[0.7,1,0.7]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif j == 0:
##                        if (carte[i-1:i+2,:2] != np.zeros((3,2))).any():
##                            ghjk = np.average(carte[i-1:i+2,:2],None,[[1,0.7],[1,1],[1,0.7]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif i == dim-1:
##                        if (carte[dim-2:,j-1:j+2] != np.zeros((2,3))).any():
##                            ghjk = np.average(carte[dim-2:,j-1:j+2],None,[[0.7,1,0.7],[1,1,1]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                    elif j == dim-1:
##                        if (carte[i-1:i+2,dim-2:] != np.zeros((3,2))).any():
##                            ghjk = np.average(carte[i-1:i+2,dim-2:],None,[[0.7,1],[1,1],[0.7,1]])+np.random.uniform(-alea,alea)
##                            if ghjk < 0:
##                                wcarte[i,j] = -1
##                            else:
##                                wcarte[i,j] = 1
##                elif (carte[i-1:i+2,j-1:j+2] != np.zeros((3,3))).any():
##                        ghjk = np.average(carte[i-1:i+2,j-1:j+2],None,[[0.7,1,0.7],[1,1,1],[0.7,1,0.7]])+np.random.uniform(-alea,alea)
##                        if ghjk < 0:
##                            wcarte[i,j] = -1
##                        else:
##                            wcarte[i,j] = 1
##        carte = np.copy(wcarte)
##    return carte

def carte_3(dim,n,sea=0.5,gini=0.8,dist_iles=None,dist_prop=None):
    '''dim: Taille de la carte. Peut être le côté de la carte en pixels (nombre
    entier strictement positif; la carte sera carrée), ou un couple contenant la
    hauteur et la longueur de la carte (tuple, liste ou ndarray de 2 nombres
    entiers strictement positifs). Valeur recommandée de quelques centaines.
    n: Nombre d\'îles. Nombre entier strictement positif.
    sea: Proportion d\'eau. Nombre compris entre 0 et 1
    gini: Indice de Gini de la répartition des surfaces entre les îles. Nombre
    compris entre 0 et 1
    dist_iles: Coefficients affectant la répartition des sources d'îles. None ou
    ndarray de taille dim contenant des nombres positifs ou nuls. Si la valeur
    est None, il y a équiprobabilité. La proportion d'eau peut ne pas être
    respectée.
    dist_prop: Zones où les îles peuvent se propager. None ou ndarray de taille
    dim contenant des 0 et des 1 (0: propagation interdite, 1: permise). N'a
    aucune influence sur le placement des sources d'îles. La proportion d'eau
    peut ne pas être respectée.'''
    if type(dim) == int:
        dim = (dim,dim)
    assert dim[0] > 0 and dim[1] > 0 and n > 0 and n <= dim[0]*dim[1] and sea >= 0 and sea <= 1 and gini >= 0 and gini <= 1, 'Les valeurs ne sont pas dans les bons intervalles.'
    if str(type(dist_prop)) == '<class \'numpy.ndarray\'>':
        assert np.shape(dist_prop) == dim and np.min(dist_prop >= 0).all() and (dist_prop <= 1).all() and (np.floor(dist_prop) == dist_prop).all(), 'Les zones de propagation permise des îles sont mal définies.'
    logging.info('Lancement de la fonction carte_3 avec les paramètres: ' + str((dim,n,sea,gini,dist_iles,dist_prop)))
    iles = np.zeros((n,3),int)
    if str(type(dist_iles)) == '<class \'NoneType\'>':
        iles[:,0] = np.random.randint(0,dim[0],n)
        iles[:,1] = np.random.randint(0,dim[1],n)
    else:
        assert np.shape(dist_iles) == dim and np.min(dist_iles) >= 0 and np.max(dist_iles) > 0, 'Les données de répartition des sources d\'îles sont incohérentes.'
        etem = np.random.choice(np.array([dim[1]*i+j for i in range(dim[0]) for j in range(dim[1])]),n,p=(dist_iles/np.sum(dist_iles)).ravel())
        iles[:,0] = etem // dim[1]
        iles[:,1] = etem % dim[1]
    iles[:,2] = dim[0]*dim[1]*(1-sea)*np.array([((i+1)/n)**(2/(1-gini)-1)-(i/n)**(2/(1-gini)-1) for i in range(n)])
    carte = np.zeros(dim,np.int8)
    for i in range(n):
        carte[iles[i,0],iles[i,1]] = 1
        iles[i,2] = max(0,iles[i,2]-1)
    encercles = 0
    while np.shape(iles)[0] != 0 and (carte != np.ones(dim)).any():
        if encercles != 0:
            pvsm = np.sum(iles[:,2])
            p = encercles / pvsm
            for i in range(np.shape(iles)[0]):
                iles[i,2] = int((1+p)*iles[i,2])
            encercles += pvsm - np.sum(iles[:,2])
            while encercles != 0:
                r = np.random.uniform(0,np.sum(iles[:,2]))
                j = 0
                while np.sum(iles[:j+1,2]) < r:
                    j += 1
                iles[j,2] += 1
                encercles -= 1
        shf = np.array([None]*np.shape(iles)[0])
        for i in range(np.shape(iles)[0]):
            r = np.random.randint(0,np.shape(iles)[0])
            while shf[r] != None:
                r = np.random.randint(0,np.shape(iles)[0])
            shf[r] = i
        wiles = []
        for i in range(np.shape(iles)[0]):
            propag = np.zeros(4,np.int8)
            if not (iles[shf[i],1] <= 0 or iles[shf[i],0] < 0 or iles[shf[i],1] >= dim[1] or iles[shf[i],0] >= dim[0]):
                if carte[iles[shf[i],0],iles[shf[i],1]-1] == 0:
                    if str(type(dist_prop)) == '<class \'NoneType\'>':
                        propag[0] = 1
                    elif dist_prop[iles[shf[i],0],iles[shf[i],1]-1] == 1:
                        propag[0] = 1
            if not (iles[shf[i],1] < 0 or iles[shf[i],0] <= 0 or iles[shf[i],1] >= dim[1] or iles[shf[i],0] >= dim[0]):
                if carte[iles[shf[i],0]-1,iles[shf[i],1]] == 0:
                    if str(type(dist_prop)) == '<class \'NoneType\'>':
                        propag[1] = 1
                    elif dist_prop[iles[shf[i],0]-1,iles[shf[i],1]] == 1:
                        propag[1] = 1
            if not (iles[shf[i],1] < 0 or iles[shf[i],0] < 0 or iles[shf[i],1] >= dim[1]-1 or iles[shf[i],0] >= dim[0]):
                if carte[iles[shf[i],0],iles[shf[i],1]+1] == 0:
                    if str(type(dist_prop)) == '<class \'NoneType\'>':
                        propag[2] = 1
                    elif dist_prop[iles[shf[i],0],iles[shf[i],1]+1] == 1:
                        propag[2] = 1
            if not (iles[shf[i],1] < 0 or iles[shf[i],0] < 0 or iles[shf[i],1] >= dim[1] or iles[shf[i],0] >= dim[0]-1):
                if carte[iles[shf[i],0]+1,iles[shf[i],1]] == 0:
                    if str(type(dist_prop)) == '<class \'NoneType\'>':
                        propag[3] = 1
                    elif dist_prop[iles[shf[i],0]+1,iles[shf[i],1]] == 1:
                        propag[3] = 1
            v = np.zeros(4,int)
            a = np.random.random_sample(4)
            p = int(iles[i,2])
            reste = np.zeros(4,int)
            a *= propag
            nsklo = np.sum(a)
            if nsklo != 0:
                a /= nsklo
                j = 0
                while j < p - sum([int(p*a[i]) for i in range(4)]):
                    r = np.random.randint(0,4)
                    if propag[r] == 1:
                        reste[r] += 1
                        j += 1
            else:
                encercles += p
            v[0] = int(p*a[0]+reste[0])
            v[1] = int(p*a[1]+reste[1])
            v[2] = int(p*a[2]+reste[2])
            v[3] = int(p*a[3]+reste[3])
            if v[0] != 0:
                carte[iles[shf[i],0],iles[shf[i],1]-1] = 1
                if v[0] > 1:
                    wiles.append([iles[shf[i],0],iles[shf[i],1]-1,float(v[0]-1)])
            if v[1] != 0:
                carte[iles[shf[i],0]-1,iles[shf[i],1]] = 1
                if v[1] > 1:
                    wiles.append([iles[shf[i],0]-1,iles[shf[i],1],float(v[1]-1)])
            if v[2] != 0:
                carte[iles[shf[i],0],iles[shf[i],1]+1] = 1
                if v[2] > 1:
                    wiles.append([iles[shf[i],0],iles[shf[i],1]+1,float(v[2]-1)])
            if v[3] != 0:
                carte[iles[shf[i],0]+1,iles[shf[i],1]] = 1
                if v[3] > 1:
                    wiles.append([iles[shf[i],0]+1,iles[shf[i],1],float(v[3]-1)])
        iles = np.array(wiles,int)
    logging.info('Déviation par rapport à la proportion d\'eau demandée: ' + str(round(100*(np.sum(carte)/((1-sea)*dim[0]*dim[1])-1),3)) + ' %')
    return carte

def carte_4(seed,it):
    '''seed: Carte à agrandir. Liste ou ndarray contenant des 0 et des 1 de
    dimension 2 et de taille au moins égale à (3,3)
    it: Nombre d\'itérations. La taille finale sera np(shape(seed))*5**it. Nombre
    entier strictement positif.
    Il est recommandé que ces deux paramètres soient tels que la taille finale
    soit inférieure à (1000,1000)'''
    assert len(np.shape(seed)) == 2 and np.shape(seed)[0] >= 3 and np.shape(seed)[1] >= 3 and it > 0, 'Les valeurs ne sont pas dans les bons intervalles.'
    logging.info('Lancement de la fonction carte_4 sur la carte ' + str(seed) + ' en ' + str(it)+ ' itérations.')
    influence_no = np.array([[int(max(7-((i+3)**2+(j+3)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_ne = np.array([[int(max(7-((i+3)**2+(j-7)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_se = np.array([[int(max(7-((i-7)**2+(j-7)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_so = np.array([[int(max(7-((i-7)**2+(j+3)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_n = np.array([[int(max(7-((i+3)**2+(j-2)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_s = np.array([[int(max(7-((i-7)**2+(j-2)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_o = np.array([[int(max(7-((i-2)**2+(j+3)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_e = np.array([[int(max(7-((i-2)**2+(j-7)**2)**0.5,0)) for j in range(5)] for i in range(5)],np.int8)
    influence_c = np.array([[int(7-((i-2)**2+(j-2)**2)**0.5) for j in range(5)] for i in range(5)],np.int8)
    influence_c[2,2] = 9
    carte = np.array(seed,dtype=np.int8,copy=True)
    for k in range(it):
        wcarte = np.zeros((5*np.shape(carte)[0],5*np.shape(carte)[1]),np.int8)
        for i in range(np.shape(carte)[0]):
            for j in range(np.shape(carte)[1]):
                v = np.zeros((3,3),np.int8)
                if i == 0:
                    if j == 0:
                        v[1:,1:] = 2 * carte[i:i+2,j:j+2] - 1
                    elif j == np.shape(carte)[1]-1:
                        v[1:,:2] = 2 * carte[i:i+2,j-1:j+1] - 1
                    else:
                        v[1:,:] = 2 * carte[i:i+2,j-1:j+2] - 1
                elif i == np.shape(carte)[0]-1:
                    if j == 0:
                        v[:2,1:] = 2 * carte[i-1:i+1,j:j+2] - 1
                    elif j == np.shape(carte)[1]-1:
                        v[:2,:2] = 2 * carte[i-1:i+1,j-1:j+1] - 1
                    else:
                        v[:2,:] = 2 * carte[i-1:i+1,j-1:j+2] - 1
                else:
                    if j == 0:
                        v[:,1:] = 2 * carte[i-1:i+2,j:j+2] - 1
                    elif j == np.shape(carte)[1]-1:
                        v[:,:2] = 2 * carte[i-1:i+2,j-1:j+1] - 1
                    else:
                        v[:,:] = 2 * carte[i-1:i+2,j-1:j+2] - 1
                trnsfr = np.sign(v[0,0] * influence_no + v[0,1] * influence_n + v[0,2] * influence_ne + v[1,0] * influence_o + v[1,1] * influence_c + v[1,2] * influence_e + v[2,0] * influence_so + v[2,1] * influence_s + v[2,2] * influence_se,dtype=np.float64)
                if 0 in trnsfr:
                    trnsfr += 0.5 * np.random.uniform(-1,1,(5,5))
                    trnsfr = np.sign(trnsfr)
                wcarte[5*i:5*(i+1),5*j:5*(j+1)] = (trnsfr + 1) / 2
        carte = np.copy(wcarte)
    return carte