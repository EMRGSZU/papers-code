import cv2
import numpy as np
from image_utils import rgb2ycbcr, ycbcr2rgb


def image_padding(img):
    block_size = 32
    hei, wid = img.shape
    hei_blk = hei // 32
    wid_blk = wid // 32

    pad_img = img[:hei_blk * 32, :wid_blk * 32]

    return pad_img


a = []
b = []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


img = cv2.imread(r'F:\desk\algoriths\data\test_set\DIV2K_valid_HR\0846.png')

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)

img = image_padding(rgb2ycbcr(img))
img = img.reshape((img.shape[0], img.shape[1], 1))
img = np.concatenate((img, img, img), 2)

# imgreconnet = cv2.imread(r'C:\Users\dell\Desktop\images\reconnet\0.1\DIV2K_valid_HR\0846.png_24.459822163664807_0.7341450650695179.png')
# imgdrnet = cv2.imread(r'C:\Users\dell\Desktop\images\dr2net\0.1\DIV2K_valid_HR\0846.png_25.109870693750512_0.7600709702078265.png')
# imgista = cv2.imread(r'C:\Users\dell\Desktop\images\istanet\DIV2K_valid_HR\0.1\0846.png_27.551403641586084_0.8179375481152973.png')
# imgcsnet = cv2.imread(r'C:\Users\dell\Desktop\images\csnet\0.1\DIV2K_valid_HR\0846_28.74540822405356_0.8538429341530037.png')
# imgcsrn = cv2.imread(r'C:\Users\dell\Desktop\images\csrn\0.1\DIV2K_valid_HR\0846_29.453201618473955_0.8643844378977729.png')

x = 820
y = 945
name = '0846111'

crop_img = img[y: y + 70, x: x + 150]
# crop_reconnet = imgreconnet[y: y+70, x: x+150]
# crop_drnet = imgdrnet[y: y+70, x: x+150]
# crop_ista = imgista[y: y+70, x: x+150]
# crop_csnet = imgcsnet[y: y+70, x: x+150]
# crop_csrn = imgcsrn[y: y+70, x: x+150]


cv2.imwrite('./{}.jpg'.format(name), crop_img)
# cv2.imwrite('./{}_reconnet.jpg'.format(name), crop_reconnet)
# cv2.imwrite('./{}_drnet.jpg'.format(name), crop_drnet)
# cv2.imwrite('./{}_ista.jpg'.format(name), crop_ista)
# cv2.imwrite('./{}_csnet.jpg'.format(name), crop_csnet)
# cv2.imwrite('./{}_csrn.jpg'.format(name), crop_csrn)

cv2.rectangle(img, (x, y), (x + 150, y + 70), (0, 0, 255), 2)
# cv2.rectangle(imgdrnet, (x, y), (x+150, y+70),(0, 0, 255), 2)
# cv2.rectangle(imgreconnet, (x, y), (x+150, y+70),(0, 0, 255), 2)
# cv2.rectangle(imgista, (x, y), (x+150, y+70),(0, 0, 255), 2)
# cv2.rectangle(imgcsnet, (x, y), (x+150, y+70),(0, 0, 255), 2)
# cv2.rectangle(imgcsrn, (x, y), (x+150, y+70),(0, 0, 255), 2)


cv2.imwrite('./{}rec.jpg'.format(name), img)
# cv2.imwrite('./{}imgreconnet.jpg'.format(name), imgreconnet)
# cv2.imwrite('./{}drnet.jpg'.format(name), imgdrnet)
# cv2.imwrite('./{}imgista.jpg'.format(name), imgista)
# cv2.imwrite('./{}imgcsnet.jpg'.format(name), imgcsnet)
# cv2.imwrite('./{}imgcsrn.jpg'.format(name), imgcsrn)