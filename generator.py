import requests
from bs4 import BeautifulSoup

url = 'https://www.instagram.com/web/search/topsearch/?context=blended&query=%23'
search_tag = str(input('Please enter the hashtag you would like to search: '))

if '#' in search_tag:
    search_tag = search_tag.strip('#')

r = requests.get(url + search_tag)
response = r.json()['hashtags']

for data in response:
    print(f"#{data['hashtag']['name']}")

import emoji

ht1 ='#' + response[-6]['hashtag']['name'] + emoji.emojize(":grinning_face_with_big_eyes:")
print("Last name : ", ht1)

# import required classes

from PIL import Image, ImageDraw, ImageFont

# create Image object with the input image

image = Image.open('beach.png')

# initialise the drawing context with
# the image object as background

draw = ImageDraw.Draw(image)

# create font object with the font file and specify
# desired size
def draw_rotated_text(image, angle, xy, text, fill, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:

    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    # get the size of our image
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim*8, max_dim*8),
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

# dimensions of image
filePath = "beach.png"
img = Image.open(filePath)
w1, h1 = img.size
h2 = int(h1*0.5)
w2 = int(w1/3)

import random
angle = random.randint(0,75)
width = random.randint(0,w2)
height = random.randint(h2,h1)

# draw the text
font = ImageFont.truetype('Roboto-Bold.ttf', 100)
draw_rotated_text(image, angle, (400, 500), ht1, (0,0,0), font=font)

image.show()

# save the edited image

image.save('new.png')