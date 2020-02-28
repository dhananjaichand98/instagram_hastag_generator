import csv
import boto3
import requests
import random
from bs4 import BeautifulSoup


with open('credentials.csv', 'r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]
photo = 'testerhere.jpg'

client = boto3.client('rekognition',
                      aws_access_key_id = access_key_id,
                      aws_secret_access_key = secret_access_key,
                      region_name = 'eu-west-1')

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

response = client.detect_labels(Image={'Bytes': source_bytes},
                                MaxLabels = 10)
labels = response['Labels']
output = [dicts for dicts in labels if(dicts['Parents'] == [])]
final = []
for dict in output:
    x = dict['Name']
    y = dict['Confidence']
    final.append([x,y])
print(final)

url = 'https://www.instagram.com/web/search/topsearch/?context=blended&query=%23'
#search_tag = str(input('Please enter the hashtag you would like to search: '))

for x in final:
    if '#' in x[0]:
        x[0] = x[0].strip('#')

for x in final:
    r = requests.get(url + x[0])
    response = r.json()['hashtags']
    print ("FOR : ",x[0])
    i = 0
    for data in response:
        if(i>5):
            break
        outputfinal = random.choice(data)
        print(f"#{data['hashtag']['name']}")
        i = i+1


