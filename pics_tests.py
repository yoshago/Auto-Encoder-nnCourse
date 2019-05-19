import numpy as np
from PIL import Image



def rgb2gray(name):

    image = Image.open(name)
    rgb = np.array(list(image.getdata())).reshape(512,512,3)
    print(rgb.shape)
    r = rgb[:,:,0]
    g =rgb[:,:,1]
    b=rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


img = rgb2gray('2.jpeg')
print(np.array(img).shape)
img=np.array(img).astype(np.uint8)
img=Image.fromarray(img)
img.show()
img.save('kineret.jpg')




