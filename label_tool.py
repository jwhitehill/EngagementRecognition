import matplotlib.pyplot as plt

badIndices = [36, 50, 54, 109, 128, 129, 141, 159, 198] + \
             [205, 206, 210, 355, 368, 374, 405, 521, 585, 590, 599, 609, 632, 647, 693, 744, 747, 759, 932, 1014, 1049, 1057] + \
	     [1140, 1160, 1230, 1265, 1274, 1426, 1480, 1497, 1684, 1711, 1743, 1765, 1787, 1839, 1840, 1848, 1856, 1932, 2006, 2017] + \
	     [2049, 2234, 2257, 2291, 2300, 2301, 2372, 2375, 2377, 2419, 2422, 2448, 2520, 2651, 2686, 2729, 2805, 2836, 2842, 2844, 2848, 2976, 2980, 3007]
idx = 2018
badIndices = []

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
