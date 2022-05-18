#discovery: this read file works when not in environment.. don't know why
#jugaad: pickle out box data separately.. not worked.. so try csv.. prob.. has \n in arrays
#try json..fails
#try npy.. error
#better not to dump, just load this every time

#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import pickle
import json

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '/home/cvpr/jayant/trainval_36/rcnn_36.tsv'




if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    iii = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            in_data[item['image_id']] = item['boxes']			#just considering boxes
            if iii%1000 == 0:
                print(iii)
            iii+=1
            #break
    #print(in_data)

    #with open('boxes36.pickle', 'wb') as handle:
    #    pickle.dump(in_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# open file for writing, "w" is writing
    #w = csv.writer(open("boxes36.csv", "w"))

# loop over dictionary keys and values
    #for key, val in in_data.items():
    # write every key and value to file
    #    w.writerow([key, val])



# Serialize data into file:
    #json.dump( in_data, open( "boxes36.json", 'w' ) )
    #print("done..!")

    #np.save('boxes36.npy',  in_data)