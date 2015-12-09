from action.data.skeletons import Skeletons
from action.data.depth_io import DepthMapBinFileIO
import os
import os.path
import sys
import math

class data_manager(object):
    def __init__(self):
        self.skeletons_clips = []
        self.dmm_clips = []

    def load(self, path):
        filename = os.path.basename(path)
        if "skeleton" in filename:
            print "loading", path
            clip = []
            label = int(filename.split('_')[0][1:])
            f = open(path)
            line = f.readline()
            nf = int(line.split(" ")[0]) #num of frames
            for i in xrange(nf):
                line = f.readline()
                if int(line) != 40:
                    for i in xrange(int(line)):
                        line = f.readline()
                    continue
                skeleton = []
                valid = True
                for j in xrange(20):
                    line = f.readline().strip()
                    x, y, z, b = line.split(' ')
                    if int(b) == 0:
                        valid = False
                    skeleton.append((float(x), float(y), float(z)))
                    line = f.readline()
                if valid:
                    clip.append(Skeletons(skeleton, label))
            self.skeletons_clips.append(clip)
        if "depth" in  filename:
            print "loading", path
            io = DepthMapBinFileIO()
            io.loadDmsFromBin("F:/data/dms/"+filename[:11])
            self.dmm_clips.append(io.DMM())

            ################# temp #################
            '''
            os.mkdir("F:/data/dms/"+filename[:11])
            io.saveDms("F:/data/dms/"+filename[:11])
            '''
            ################# temp #################

            # self.depth_clips.append(io.dms)

    def load_all(self, path):
        for parent, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                self.load_all(os.path.join(parent, dirname))
            for filename in filenames:
                self.load(os.path.join(parent, filename))

    def gen_set(self, fed, fbc = 0, win = 1):
        '''
        fed: feature nums of euclid distance
        fmc: feature nums of border coordinates 
        '''
        assert(fed <= 190)

        features_clips = []
        labels_clips = []
        for clip in self.skeletons_clips:
            features = []
            labels = []
            for skeleton in clip:
                items = skeleton.ed.items()
                backitems = [[v[1], v[0]] for v in items]
                backitems.sort()
                features.append([backitems[i][1] for i in xrange(fed)]) #euclid distance feature
                labels.append(skeleton.label)
            features_clips.append(features)
            labels_clips.append(labels)

        return features_clips, labels_clips

    def gen_set_full(self, sparse = 10):
        features_clips = []
        labels_clips = []
        for c in xrange(len(self.skeletons_clips)):
            clip = self.skeletons_clips[c]
            features = []
            labels = []

            sparse_nums = len(clip) / sparse
            start_i = (len(clip) - sparse_nums*sparse)/2
            prev_sk = None
            for f in xrange(sparse_nums):
                temp_vec = [0] * 270
                items = clip[start_i + f*sparse].ed.items()
                backitems = [[v[1], v[0]] for v in items]
                backitems.sort()
                for i in xrange(190):
                    temp_vec[backitems[i][1]] = i #euclid distance orderlet feature

                backitems = [[clip[start_i + f*sparse].csk[i][0], i] for i in xrange(len(clip[start_i + f*sparse].csk))]
                backitems.sort()
                for i in xrange(20):
                    temp_vec[backitems[i][1]+190] = int((i+1)*9.5) #border coordinates orderlet feature
                backitems = [[clip[start_i + f*sparse].csk[i][1], i] for i in xrange(len(clip[start_i + f*sparse].csk))]
                backitems.sort()
                for i in xrange(20):
                    temp_vec[backitems[i][1]+210] = int((i+1)*9.5) #border coordinates orderlet feature
                backitems = [[clip[start_i + f*sparse].csk[i][2], i] for i in xrange(len(clip[start_i + f*sparse].csk))]
                backitems.sort()
                for i in xrange(20):
                    temp_vec[backitems[i][1]+230] = int((i+1)*9.5) #border coordinates orderlet feature

                if prev_sk is None:
                    prev_sk = clip[start_i + f*sparse]
                    continue
                    # backitems = [[0, i] for i in xrange(len(clip[start_i + f*sparse].csk))]
                else:
                    backitems = [[ \
                        math.sqrt( \
                        (clip[start_i + f*sparse].csk[i][0] - prev_sk.csk[i][0])**2 + \
                        (clip[start_i + f*sparse].csk[i][1] - prev_sk.csk[i][1])**2 + \
                        (clip[start_i + f*sparse].csk[i][2] - prev_sk.csk[i][2])**2) \
                    , i] for i in xrange(len(clip[start_i + f*sparse].csk))]
                backitems.sort()
                for i in xrange(20):
                    temp_vec[backitems[i][1]+250] = int((i+1)*9.5) #motion length orderlet feature
                prev_sk = clip[start_i + f*sparse]

                temp_vec += list(self.dmm_clips[c][0, f-1].flatten())
                temp_vec += list(self.dmm_clips[c][1, f-1].flatten())
                temp_vec += list(self.dmm_clips[c][2, f-1].flatten())

                features.append(temp_vec)
                labels.append(clip[start_i + f*sparse].label)
            features_clips.append(features)
            labels_clips.append(labels)
        return features_clips, labels_clips

# if __name__ == '__main__':
#     dm = data_manager()
#     dm.load_all("F:/data/S2_T")
#     train_lex, train_y = dm.gen_set_full()
#     print train_lex[0][0]