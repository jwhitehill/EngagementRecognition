import numpy as np

def makeGaborKernel (n, freq, orient, bwFreq, bwOrient):
	powBwFreq = 2 ** bwFreq
	C = (np.pi/np.log(2)) ** 0.5
	a = C*(powBwFreq-1.)/(powBwFreq+1.)*freq
	b = C*np.tan(bwOrient/2)*freq
	kNorm = a*b

	kernel = np.zeros((n, n), dtype=np.complex64)
	for r in range(n):
		for c in range(n):
			x = (np.cos(orient)*(c-(n-1)/2.) + np.sin(orient)*(r-(n-1)/2.))
			y = (-np.sin(orient)*(c-(n-1)/2.) + np.cos(orient)*(r-(n-1)/2.))
			ux = freq * x
			kernel[r,c] = np.exp(-np.pi*(x*x*a*a+y*y*b*b) + 2*np.pi*ux*1j)
	return kernel


def makeGaborFilterBank (n):
	bank = []
	for f in range(5):
		R = 0.75 * pow(2, 1.4)
		freq = 8./(R ** f) * 4./n
		for orient in np.arange(0, np.pi, np.pi/8):
			bank.append(makeGaborKernel(n, freq, orient, 1.4, 40*np.pi/180))
	return bank

if __name__ == "__main__":
	bank = makeGaborFilterBank(32)
	bankF = np.fft.fft2(bank)
