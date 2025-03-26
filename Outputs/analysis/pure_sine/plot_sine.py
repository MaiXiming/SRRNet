import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 1, 0.01)

lw = 5
for hii in range(5):
    harmonic = hii + 1
    y1 = np.sin(2*np.pi*2*harmonic*t)
    y2 = np.cos(2*np.pi*2*harmonic*t)
    plt.figure(figsize=(10,5))
    plt.plot(t, y1, 'k', linewidth=lw)
    plt.axis('off')
    plt.savefig(f'sines/sine{harmonic}.png')

    plt.figure(figsize=(10,5))
    plt.plot(t, y2, 'k', linewidth=lw)
    plt.axis('off')
    plt.savefig(f'sines/cosine{harmonic}.png')