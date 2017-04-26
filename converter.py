import os
import timeit
from pydub import AudioSegment
from utils import DATASET_DIR

start = timeit.default_timer()
root_dir = DATASET_DIR
print "START CONVERT DATASET"
for subdir, dirs, files in os.walk(root_dir):
    for fl in files:
        print "File: ", fl
        path = subdir+'/'+fl
        if path.endswith("au"):
            song = AudioSegment.from_file(path, "au")
            song = song[:30000]
            song.export(path[:-2]+"wav", format='wav')

for subdir, dirs, files in os.walk(root_dir):
    for fl in files:
        path = subdir+'/'+fl
        if not path.endswith("wav"):
            os.remove(path)

stop = timeit.default_timer()
print "Conversion time = ", (stop - start) 
