import numpy as np

from PIL import Image
# Save image to the file
def Save(ar, d="out"):
    ar_ = np.zeros(ar.shape, dtype='uint8')
    ar_ = (ar * 255).astype('uint8')
    im = Image.fromarray(ar_)
    im.save(d+".jpeg")

# Normalize the image
# min value and max value, shift are not working though...
def Normalize(ar, minV = 0, maxV = 255, shift = 0):
    ar_ = np.zeros(ar.shape, dtype='float64')
    ar_ = ar / 255
    return ar_

# Add Normalized Random Noise
def AddNoiseNormal(ar, amount = 0.02, median = 0.0, sigma = 1.0):
    ar_ = np.random.normal(median, sigma, ar.shape)
    return ar + ar_ * amount

# Add Random Noise
def AddNoise(ar, amount = 0.1):
    ar_ = np.random.random(ar.shape)
    return (ar + ar_ * amount) - (amount / 2)

# Color Jittering
def ColorJitter(ar):
    return ar

# Rotate image
def Rotate(ar, angle = 0):
    return ar

# Translate image
def Translate(ar):
    return ar

# Flipping image
def Flip(ar, hori = False):
    return ar