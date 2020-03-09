pass
import csv
import boto3
import requests
import random
from bs4 import BeautifulSoup
import extracto
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import emoji
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
import sys
# %matplotlib inline
from PIL import Image, ImageDraw, ImageFont

from emo_uni import emo_list, emo_get

photo = 'dream.jpg'

with open('credentials.csv', 'r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

client = boto3.client('rekognition',
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_access_key,
                      region_name='eu-west-1')

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

response = client.detect_labels(Image={'Bytes': source_bytes}, MaxLabels=10)

labels = response['Labels']
output = [dicts for dicts in labels if (dicts['Parents'] == [])]
final = []
for dict in output:
    x = dict['Name']
    y = dict['Confidence']
    final.append([x, y])
print(final)

url = 'https://www.instagram.com/web/search/topsearch/?context=blended&query=%23'
# search_tag = str(input('Please enter the hashtag you would like to search: '))

for x in final:
    if '#' in x[0]:
        x[0] = x[0].strip('#')

keyword = final[0][0]
r = requests.get(url + keyword)
response = r.json()['hashtags']
print("FOR : ", keyword)
i = 0

options = [2, 3, 4, 5, 6, 7]
chosenval = random.choice(options)
message = '#'
message += (response[chosenval]['hashtag']['name'])
print(response[chosenval]['hashtag']['name'])

# for data in response:
#     if(i>5):
#         break
#
#
#     #outputfinal = random.choice(data)
#
#     print(f"#{data['hashtag']['name']}")
#     i = i+1
len(emo_list)

z = emo_list[2390]

print(z)

e_l = []
for i in emo_list:
    e_l.append(str(i.replace("_", " ")).lower())

e_l[1:10]

print(emoji.emojize(':India: is the greatest'))

import spacy

nlp = spacy.load('en')
from tqdm import tqdm

with open('glove.6B.300d.txt', 'r', encoding="utf8") as f:
    for line in tqdm(f, total=400000):
        parts = line.split()
        word = parts[0]
        vec = np.array([float(v) for v in parts[1:]], dtype='f')
        nlp.vocab.set_vector(word, vec)

docs = [nlp(str(keywords)) for keywords in tqdm(e_l)]
doc_vectors = np.array([doc.vector for doc in docs])

from numpy import dot
from numpy.linalg import norm


def most_similar(vectors, vec):
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))
    return (np.argsort(-dst))[0], max(dst)


sentences = [keyword]
emo = []
print("keyword = ", keyword)
for sentence in sentences:
    l = []
    for w in sentence.split(" "):
        v = nlp(w.lower()).vector
        ms, sim = most_similar(doc_vectors, v)
        # print(sim)
        if (sim > 0.0115):
            word = emo_get[ms]
            l.append(emoji.emojize(word, use_aliases=True))
        else:
            l.append(w)
    print(sentence)

    for x in l:
        emo.append(x)
        print(x)
    display(HTML('<font size="+3">{}</font>'.format(' '.join([x for x in l]))))

emo1 = emo[0]

dom_color = extracto.get_colors(photo)
print(dom_color[0])
r, g, b = dom_color[0]

from colorsys import rgb_to_hsv, hsv_to_rgb


def complementary(r, g, b):
    # returns RGB components of complementary color
    hsv = rgb_to_hsv(r, g, b)
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])


print(255 - r, 255 - g, 255 - b)
texter = (255 - r, 255 - g, 255 - b)
alpha = 0.4


# s_img = cv2.imread("smaller_image.png")
# l_img = cv2.imread("larger_image.jpg")
# x_offset=y_offset=50
# photo[y_offset, x_offset] = imaging
# imaging.copyTo(photo(cv::Rect(400,40, imaging.cols, imaging.rows)))
class ImageMetaData(object):
    exif_data = None
    image = None

    def __init__(self, img_path):
        self.image = Image.open(img_path)

        self.get_exif_data()
        super(ImageMetaData, self).__init__()

    def get_exif_data(self):
        exif_data = {}
        info = self.image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        self.exif_data = exif_data
        return exif_data

    def get_if_exist(self, data, key):
        if key in data:
            return data[key]
        return None

    def convert_to_degress(self, value):
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        return d + (m / 60.0) + (s / 3600.0)

    def get_lat_lng(self):
        lat = None
        lng = None
        exif_data = self.get_exif_data()

        if "GPSInfo" in exif_data:
            gps_info = exif_data["GPSInfo"]
            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = self.get_if_exist(gps_info, 'GPSLatitudeRef')
            gps_longitude = self.get_if_exist(gps_info, 'GPSLongitude')
            gps_longitude_ref = self.get_if_exist(gps_info, 'GPSLongitudeRef')
            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":
                    lat = 0 - lat
                lng = self.convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = 0 - lng
        return lat, lng

    def getlocation(self, lat, long):
        URL = "https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=" + str(
            latlng[0]) + "&longitude=" + str(latlng[1]) + "&localityLanguage=en"
        r = requests.get(url=URL, params=None)
        data = r.json()
        return data['principalSubdivision']


imagefile = photo
meta_data = ImageMetaData(imagefile)
latlng = meta_data.get_lat_lng()
# print(latlng)
exif_data = meta_data.get_exif_data()
loc = '@' + meta_data.getlocation(latlng[0], latlng[1])
print(loc)

print('loc =', loc, 'hashtag =', message)

"""
while True:
    image = cv2.imread(photo)
    cv2.rectangle(image, (0, 0), (300, 30), texter, -1)
    cv2.putText(image, message, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.rectangle(image, (0, 70), (200, 100), texter, -1)
    cv2.putText(image, loc, (40, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.imshow('labeled', image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
"""

# code for texts starts

import requests

# initialise the drawing context with
# the image object as background
image = Image.open('dream.jpg')
draw = ImageDraw.Draw(image)


# create font object with the font file and specify
# desired size
def draw_rotated_text(image, angle, xy, text, fill, *args, **kwargs):
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, thickness=5, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim * 8, max_dim * 8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
            mask_size, resample=Image.LANCZOS)

    # crop the mask to match image
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)

    # paste the appropriate color, with the text transparency mask
    color_image = Image.new('RGBA', image.size, fill)
    image.paste(color_image, mask)


width, height = image.size
# for the emoji
font_color = (0, 0, 0)
font_size = 72
unicode_text = emoji.emojize(emo1)
unicode_font = ImageFont.truetype("Symbola.ttf", font_size)
draw.text((100, 50), unicode_text, font=unicode_font, fill=font_color)

# for the hashtag
font = ImageFont.truetype('Cosmopolitan Sans Bold.otf', 100)
draw_rotated_text(image, 20, (int(width / 12), int(height / 1.5)), message, (255, 255, 255), font=font)

# for the location
font1 = ImageFont.truetype('arial.ttf', 30)
draw_rotated_text(image, 0, (width - 200, 10), loc, (255, 255, 255), font=font1)

image.show()
# save the edited image
image.save('new.png')