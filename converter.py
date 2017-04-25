import os
import sys
import timeit

from pydub import AudioSegment
from utils import GENRE_DIR

start = timeit.default_timer()
rootdir = GENRE_DIR
print "START CONVERT DATASET"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print "File: " ,file 
        path = subdir+'/'+file
        if path.endswith("au"):
            song = AudioSegment.from_file(path,"au")
            song = song[:30000]
            song.export(path[:-2]+"wav",format='wav')

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = subdir+'/'+file
        if not path.endswith("wav"):
            os.remove(path)

stop = timeit.default_timer()
print "Conversion time = ", (stop - start) 
