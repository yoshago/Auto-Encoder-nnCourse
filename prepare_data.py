from PIL import Image
import numpy as np


SQUARE_SIZE=16


def my_print(single, size):
    to_show=[]
    for k in range(size):
        line=[]
        for l in range(size):
            line.append(single[size*k+l])
        to_show.append(line)
    to_show=np.array(to_show).astype(np.uint8)

#    img = Image.fromarray(to_show)
#    img.show()
    return to_show



def prepare_data(name,jump_size):
    image = Image.open(name)
    image=np.array(list(image.getdata())).reshape(512,512)

    data_set=[]
    for i in range(len(image)-15):
        if(jump_size*i>=(512-SQUARE_SIZE)):
            break
        for j in range(len(image)-15):
            line=[]
            if(jump_size*j>=(512-SQUARE_SIZE) ):
                break
            for k in range(SQUARE_SIZE):
                for l in  range(SQUARE_SIZE):
                    line.append(image[jump_size*i+k][jump_size*j+l])
            data_set.append(line)
    return data_set


def img_to_data(name):
    image = Image.open(name)
    image = np.array(list(image.getdata())).reshape(512, 512)
    data_set=[]
    for i in range(32):
        if (SQUARE_SIZE * i >= (512 - SQUARE_SIZE+1)):
            break
        for j in range(32):
            if (SQUARE_SIZE * j >= (512 - SQUARE_SIZE+1)):
                break
            line = []
            for k in range(SQUARE_SIZE):
                for l in  range(SQUARE_SIZE):
                    line.append(image[SQUARE_SIZE*i+k][SQUARE_SIZE*j+l])
            data_set.append(line)
    return data_set

def text_to_data(text):
    image = Image.frombytes('L', (512,512), bytes(text,'utf-8'))
    image = np.array(list(image.getdata())).reshape(512, 512)
    data_set=[]
    for i in range(32):
        if (SQUARE_SIZE * i >= (512 - SQUARE_SIZE+1)):
            break
        for j in range(32):
            if (SQUARE_SIZE * j >= (512 - SQUARE_SIZE+1)):
                break
            line = []
            for k in range(SQUARE_SIZE):
                for l in  range(SQUARE_SIZE):
                    line.append(image[SQUARE_SIZE*i+k][SQUARE_SIZE*j+l])
            data_set.append(line)
    return data_set

def data_to_img(data):
    print(len(data))
    print(len(data[1]))
    img=np.zeros((512,512))
    for i in range(32):
        for j in range(32):
            to_show=my_print(data[32*i+j],16)
            for k in range(16):
                for l in range(16):
                    img[16*i+k][16*j+l]=int(to_show[k][l])


    img=img.astype(np.uint8)
    img=Image.fromarray(img)
    img.show()
    return img

def save_data(name, data):
    to_save=Image.fromarray(np.array(data).astype(np.uint8))
    to_save.save(name)
#data_to_img(img_to_data("Lena.jpg"))
