import math

class Skeletons(object):
    def __init__(self, csk, label):
        ''' 
        csk: spatial coordinates of skeleton
        ed: euclid distance
        '''
        self.csk = csk
        self.label = label

        self.ed = dict()
        count = 0 
        for i in xrange(20):
            for j in xrange(i+1, 20):
                self.ed[count] = math.sqrt( \
                    (csk[i][0] - csk[j][0])**2 + \
                    (csk[i][1] - csk[j][1])**2 + \
                    (csk[i][2] - csk[j][2])**2)
                count += 1