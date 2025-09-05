# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:06:31 2021
RANSAC FOR RECTS AND CIRCUMFERENCES
Random sample consensus is an iterative method to estimate parameters 
of a mathematical model from a set of observed data that contains outliers
@author: eusebio
"""

import numpy as np
import lms
import points as pts

def inliers_recta( ptos_cnt,A,B,C,tolerancia):
    '''
    Select in ptos_cnt the inliers the rect Ax2+Bx+C=0
    It supposes n(A,B) is unitary to compute de distance without divinding 
    dist(P,r) = abs(A*x + B*y + C)
    normAB=sqrt(A^2 + B^2)=1

    Parameters
    ----------
    ptos_cnt : TYPE 3D array [x,y]
        DESCRIPTION. List with all contour points (x,y)
    A : TYPE float
        DESCRIPTION. value of A in Ax2+Bx+C=0
    B : TYPE float
        DESCRIPTION. value of B in Ax2+Bx+C=0
    C : TYPE float
        DESCRIPTION. value of C in Ax2+Bx+C=0
    tolerancia : TYPE int
        DESCRIPTION. acceptable distance in pixels to the rect to be consider inlier

    Returns
    -------
    TYPE np.array with the coordinates of inliers
        DESCRIPTION.array wit the coordinates of inliers for the rect Ax2+Bx+C=0

    '''
    
    inliers = []
    num_ptos = len(ptos_cnt)
    num_inliers = 0
    for k in range(num_ptos):
        x = ptos_cnt[k][0][0]
        y = ptos_cnt[k][0][1]
        dist_punto_recta = abs(A*x + B*y + C);  
        if  dist_punto_recta < tolerancia:  
            num_inliers = num_inliers + 1;
            inliers.append( ptos_cnt[k] )
    
    return np.array(inliers)

def inliers_circunf( ptos_cnt,A,B,C,tolerancia):
    '''
    Return the inliers in ptos_cnt for the circumf x2+y2+ Ax+By+C=0 with 
    tolerance tolerancia

Parameters
    ----------
    ptos_cnt : TYPE 3D array [x,y]
        DESCRIPTION. List with all contour points (x,y)
    A : TYPE float
        DESCRIPTION. value of A in x2+y2+Ax2+Bx+C=0
    B : TYPE float
        DESCRIPTION. value of B in x2+y2+Ax2+Bx+C=0
    C : TYPE float
        DESCRIPTION. value of C in x2+y2+Ax2+Bx+C=0
    tolerancia : TYPE int
        DESCRIPTION. acceptable distance in pixels to the circumf to be consider inlier

    Returns
    -------
    TYPE np.array with the coordinates of inliers
        DESCRIPTION.array wit the coordinates of inliers for the rect x2+y2+Ax2+Bx+C=0

    '''
    radio = np.sqrt(A**2+B**2-4*C)/2
    centro = ((-A/2), int(-B/2))
    #consensus interval (R1,R2)
    R1 = radio-tolerancia
    R2 = radio+tolerancia
    inliers = []
    num_ptos = len(ptos_cnt)
    for k in range(num_ptos):
        x = ptos_cnt[k][0][0]
        y = ptos_cnt[k][0][1]
        dist_centro = np.sqrt((x-centro[0])**2 + (y-centro[1])**2)
        if R1 < dist_centro < R2:
            inliers.append(ptos_cnt[k])
    return np.array(inliers)

def ransac_recta(puntos,num_max_iter):
    '''

    Parameters
    ----------
    puntos : TYPE 3D array [x,y]
        DESCRIPTION. array with all contour points coordinates (x,y)
    num_max_iter : TYPE integer
        DESCRIPTION. Maximum number of iterations

    Returns
    -------
    best_inliers : TYPE array
        DESCRIPTION. The inliers for the best model found

    '''

    tolerancia=1; #maximum distance to the rect 
    num_inliers=0;
    mejor_num_inliers=0; #maximum number of inliers found

    for i in range(num_max_iter):
        samples = pts.get_random_points(puntos, 2)
        #rect defined by these two points
        [A, B, C] = lms.recta(samples)
        inliers = inliers_recta( puntos, A,B,C, tolerancia)

        num_inliers=len(inliers);
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers=num_inliers
            best_inliers=inliers

    return best_inliers
def ransac_circunf(puntos, num_max_iter):
    '''

    Parameters
    ----------
    puntos : TYPE 3D array [x,y]
        DESCRIPTION. array with all contour points coordinates (x,y)
    num_max_iter : TYPE integer
        DESCRIPTION. Maximum number of iterations

    Returns
    -------
    best_inliers : TYPE array
        DESCRIPTION. The inliers for the best model found

    '''

    tolerancia = 1 #maximum distance to the rect
    num_inliers = 0
    mejor_num_inliers = 0 #maximum number of inliers found

    for i in range(num_max_iter):
        samples = pts.get_random_points(puntos, 3)#three point to define circumf
        #circumf. defined by these three points
        [A, B, C] = lms.circunf(samples)
        inliers = inliers_circunf(puntos, A,B,C, tolerancia)

        num_inliers = len(inliers)
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers = num_inliers
            best_inliers = inliers
    return best_inliers