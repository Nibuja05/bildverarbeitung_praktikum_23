import sys, os
import numpy as np
from math import sin
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from cv2 import filter2D, imread, imwrite
from scipy.linalg import pascal, dft

# ==================================
# 1 Fourier Transformation
# ==================================
def task_1():
	s1 = lambda x: sin(x)
	s2 = lambda x: sin(x) + (3 * sin(2 * x + 1) - 1)
	
	region = (0, 4 * np.pi)

	checkDft(s1, 20, region)
	checkDft(s2, 20, region)

def checkDft(func, rate, region):
	showFunc(func, region, rate)
	
	amount = int((region[1] / np.pi) * rate)
	xList = [(x / amount) * region[1] for x in range(amount + 1)]
	yList = np.array([func(x) for x in xList])
	
	res = manualDft(yList)
	real = np.real(res)

	# zur Überprüfung
	res2 = np.fft.fft(yList)
	real2 = np.real(res2)

	showValues(xList, yList)
	showValues(xList, real)
	# showValues(xList, real2)


def manualDft(x):
    N = np.size(x)
    X = np.zeros((N,),dtype=np.complex128)
    for m in range(0,N):    
        for n in range(0,N):
            X[m] += x[n] * np.exp(-np.pi*2j * m * n/N)
    return X
	

# ==================================
# 2 Box Filter
# ==================================
def task_2():
	image = imread("test.png")

	# a)
	kernel = np.full((5,5), 1)
	kernel = normalizeKernel(kernel)
	result = filter2D(image, -1, kernel)
	imwrite("result_box.png", result)

	# b)
	for x in range(1, 6):
		kernel[2,2] *= 2
		kernel = normalizeKernel(kernel)
		result = filter2D(image, -1, kernel)
		imwrite("result_box_%d.png" % 2 ** x, result)

	# -> Weniger Blur, Annäherung an Original
	# da Original-Pixel größten Einfluss hat


def normalizeKernel(kernel):
	total = np.sum(kernel)
	return np.multiply(kernel, 1 / total)

# ==================================
# 3 Implementierung des Gaußfilters
# ==================================
def task_3():
	image = imread("test.png")

	# b)
	for x in [5, 7, 15]:
		kernel = gaussFilter(x)
		result = filter2D(image, -1, kernel)
		imwrite("result_gaussian_%d.png" % x, result)

def gaussFilter(size):
	triangle = pascal(size, kind="lower") # Pascal triangle
	pRow = triangle[-1] # last row
	kernel = np.outer(pRow, pRow) # last row * last row
	return normalizeKernel(kernel)


# ==================================

def showFunc(func, region, rate = 1):
	fig, ax = preparePlot()
	amount = int((region[1] / np.pi) * rate)
	xList = [(x / amount) * region[1] for x in range(amount + 1)]
	# xList = [(x / amount) * (region[1] - region[0]) + region[0] for x in range(amount + 1)]
	yList = [func(x) for x in xList]

	for x in xList:
		ax.axvline(x, color="black")
	ax.plot(xList, yList, color="red")

	plt.show()

def showArr(arr):
	fig, ax = preparePlot()
	ax.plot(arr)
	plt.show()

def showValues(xList, yList):
	fig, ax = preparePlot()
	ax.plot(xList, yList)
	plt.show()

def preparePlot():
	fig,ax = plt.subplots()

	ax.xaxis.set_major_formatter(FuncFormatter(
	lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
	))
	ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

	return fig,ax

if __name__ == "__main__":
	task_1()
	task_2()
	task_3()