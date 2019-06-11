#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:00:28 2019

@author: tangcode
"""
import numpy as np
import cv2
import urllib.request as ur
import csv

a = []
# Read and load the CSV
with open('/Users/tangcode/Downloads/images_600_plus.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # print(f'\t{row[0]} - {row[1]}')
        a.append([row[0], row[1], '', 0])

for i in range(len(a)):
    url = a[i][0]
    # Open the URL with the image and load it as an array
    resp = ur.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # calculate mean value from RGB channels and flatten to 1D array
    vals = im.mean(axis=2).flatten()
    # calculate histogram
    counts, bins = np.histogram(vals, range(257))
    bigArray = counts[np.argpartition(counts, -30)[-30:]]
    parameter = np.sum(bigArray) / np.sum(counts)
    a[i][3] = parameter
    if (parameter) > 0.50:
        a[i][2] = 's'
    else:
        a[i][2] = 'c'
# Print the URLs that have a different result in the classification
for i in range(len(a)):
    if a[i][1] != a[i][2]:
        print(a[i])
# Save the results in a response csv
with open('/Users/tangcode/Downloads/images_output_600_plus.csv', mode='w') as output_file:
    output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(a)):
        output_writer.writerow([a[i][0], a[i][1], a[i][2], a[i][3]])