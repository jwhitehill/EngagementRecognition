import matplotlib.pyplot as plt

badIndices = [36, 50, 54, 109, 128, 129, 141, 159, 198] + \
             [205, 206, 210, 355, 368, 374, 405, 521, 585, 590, 599, 609, 632, 647, 693, 744, 747, 759, 932, 1014, 1049, 1057] + \
	     [1140, 1160, 1230, 1265, 1274, 1426, 1480, 1497, 1684, 1711, 1743, 1765, 1787, 1839, 1840, 1848, 1856, 1932, 2006, 2017] + \
	     [2049, 2234, 2257, 2291, 2300, 2301, 2372, 2375, 2377, 2419, 2422, 2448, 2520, 2651, 2686, 2729, 2805, 2836, 2842, 2844, 2848, 2976, 2980, 3007] + \
	     [3007, 3127, 3124, 3202, 3321, 3338, 3389, 3399, 3474, 3439, 3581, 3591, 3584, 3703, 3746, 3806, 3834, 3855, 3872, 3918, 3926, 4031, 4087, 4091, 4106, 4146, 4202, 4203, 4240, 4302, 4317, 4379, 4395, 4440, 4465, 4597, 4612, 4709, 4807, 4825, 4827, 4838, 4884, 4953, 5008, 5037, 5076, 5104, 5211, 5249, 5291, 5313, 5330, 5397, 5421, 5449, 5486, 5497, 5532, 5533, 5584, 5647, 5665, 5672, 5674, 5736, 5741, 5761, 5800, 5846, 5854, 5902, 5982, 5996, 6034, 6106, 6152, 6162, 6200, 6303, 6336, 6341, 6454, 6549, 6567, 6652, 6653, 6656, 6663, 6687] + \
	     [6703, 6757, 6762, 6763, 6767, 6781, 6848, 6859, 6875, 6951, 7036, 7172, 7202, 7226, 7232, 7247, 7306, 7313, 7320, 7344, 7357, 7403, 7423, 7443, 7475, 7483, 7494, 7634, 7741, 7831, 7851, 7868, 7932, 7967, 7969, 7977, 8031, 8050, 8177, 8214, 8241, 8254, 8286, 8354, 8389, 8406, 8462, 8477, 8497, 8501, 8512, 8523, 8529, 8629, 8678, 8890, 8933, 8934, 9007, 9239, 9282, 9373, 9384, 9405, 9433, 9481, 9497, 9506, 9511, 9532, 9544, 9563, 9614, 9664, 9717, 9726, 9751, 9808, 9887, 9910, 9925, 9972, 9974, 10044, 10087, 10107, 10149, 10159, 10202, 10251, 10340, 10374, 10455, 10505, 10571, 10604, 10633, 10684]
idx = 10685
badIndices = []

def label_single ():
    global im, idx
    def press (event):
        sys.stdout.flush()
        global idx
        global im
        global badIndices
        if event.key == 'n':
            idx += 1
        elif event.key == 'p':
            idx -= 1
        elif event.key == 'x':
            badIndices.append(idx)
        im.set_data(faces[idx,:,:])
        fig.canvas.draw()
        print idx
    
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(faces[idx,:,:], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', press)
    fig.show()

def label_many ():
    N = 10
    global im, idx
    def key_press (event):
        sys.stdout.flush()
        global idx
        global im
        if event.key == 'n':
            idx += N*N
        elif event.key == 'p':
            idx -= N*N
        im.set_data(makeMontage(faces, range(idx, idx+N*N)))
        fig.canvas.draw()
        print idx
    def mouse_press (event):
        global idx
        r = int(event.ydata / faces.shape[1])
        c = int(event.xdata / faces.shape[1])
        clickedIdx = idx + r * N + c
        print clickedIdx
        badIndices.append(clickedIdx)
    def makeMontage (faces, idxs):
        montage = np.zeros((N * faces.shape[1], N * faces.shape[1]))
	for idx in idxs:
           r = (idx - min(idxs)) / N * faces.shape[1]
           c = (idx - min(idxs)) % N * faces.shape[1]
	   montage[r:r+faces.shape[1], c:c+faces.shape[1]] = faces[idx,:,:]
	return montage
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(makeMontage(faces, range(idx, idx+N*N)), cmap='gray')
    fig.canvas.mpl_connect('button_press_event', mouse_press)
    fig.canvas.mpl_connect('key_press_event', key_press)
    fig.show()

if __name__ == "__main__":
    label_single()
