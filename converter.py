import os
import timeit
from pydub import AudioSegment
from utils import DATASET_DIR
from utils import TEST_DIR

start = timeit.default_timer()
root_dir = TEST_DIR
print "START CONVERT DATASET"
print root_dir
for subdir, dirs, files in os.walk(root_dir):
    for fl in files:
        print "File: ", fl 
        path = subdir+'/'+fl
        if path.endswith("mp3"):
            print path
            song = AudioSegment.from_file(path, "mp3")
            song = song[60000:90000]
            song.export(path[:-3]+"wav", format='wav')

for subdir, dirs, files in os.walk(root_dir):
    for fl in files:
        path = subdir+'/'+fl
        if not path.endswith("wav"):
            os.remove(path)

stop = timeit.default_timer()
print "Conversion time = ", (stop - start) 
