import matplotlib.pyplot as plt

badIndices = [36, 50, 54, 109, 128, 129, 141, 159, 198] +
             [205, 206, 210, 355, 368, 374, 405, 521, 585, 590, 599, 609, 632, 647, 693, 744, 747, 759, 932, 1014, 1049, 1057]
idx = 0
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
