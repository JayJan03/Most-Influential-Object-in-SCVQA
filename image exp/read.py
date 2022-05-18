
#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '/home/cvpr/jayant/trainval_36/rcnn_36.tsv'


if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    iii = 0
    with open(infile, "rb") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            #item['image_id'] = int(item['image_id'])
            #item['image_h'] = int(item['image_h'])
            #item['image_w'] = int(item['image_w'])   
            #item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            in_data[int(item['image_id'])] = item
            if iii%1000 == 0:
                print(iii)
            iii+=1
            #break
    print(in_data[14450])
    #print(in_data[1])