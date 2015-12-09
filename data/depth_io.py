import numpy as np
import matplotlib.pyplot as plt
import scipy.misc, os
import struct

class DepthMapBinFileIO(object):
    
    def loadDms(self, file = "D:\\tddownload\\MSR-Action3D\\a05_s05_e01_sdepth.bin"):
        self.dms = None
        self.skeletonID = None
        if os.path.isfile(file):
            f = open(file, "rb")
            # tempbytes = bytearray(f.read(4))
            # self.frames = (tempbytes[0]&0xFF)+(tempbytes[1]<<8&0xFFFF)+(tempbytes[2]<<16&0xFFFFFF)+(tempbytes[3]<<24&0xFFFFFFFF)
            # tempbytes = bytearray(f.read(4))
            # self.cols = (tempbytes[0]&0xFF)+(tempbytes[1]<<8&0xFFFF)+(tempbytes[2]<<16&0xFFFFFF)+(tempbytes[3]<<24&0xFFFFFFFF)
            # tempbytes = bytearray(f.read(4))
            # self.rows = (tempbytes[0]&0xFF)+(tempbytes[1]<<8&0xFFFF)+(tempbytes[2]<<16&0xFFFFFF)+(tempbytes[3]<<24&0xFFFFFFFF)

            self.frames = struct.unpack('i', f.read(4))[0]
            self.cols = struct.unpack('i', f.read(4))[0]
            self.rows = struct.unpack('i', f.read(4))[0]
            
            self.dms = np.zeros((self.frames, self.rows, self.cols))
            self.skeletonID = np.zeros((self.frames, self.rows, self.cols))

            for i in xrange(self.frames):
                for r in xrange(self.rows):
                    temp = np.zeros((self.rows, self.cols))
                    for c in xrange(self.cols):
                        # tempbytes = bytearray(f.read(4))
                        # temp=(tempbytes[0]&0xFF)+(tempbytes[1]<<8&0xFFFF)+(tempbytes[2]<<16&0xFFFFFF)+(tempbytes[3]<<24&0xFFFFFFFF)
                        # self.dms[i,r,c] = int(temp)
                        # tempbytes = bytearray(f.read(1))
                        # self.skeletonID[i,r,c] = int(tempbytes[0]&0xFF)
                        temp[r, c] = struct.unpack('i', f.read(4))[0]
                    for c in xrange(self.cols):
                        self.skeletonID[i,r,c] = struct.unpack('B', f.read(1))[0]
                        if self.skeletonID[i,r,c] > 0:
                            self.dms[i,r,c] = temp[r, c]
                    # for c in xrange(self.cols):
                    #     if self.dms[i,r,c] != 0:
                    #         print self.dms[i,r,c], self.skeletonID[i,r,c]

            # self.ishape = (self.frames, self.rows, self.cols, np.max(self.dms))
            self.ishape = (self.frames, self.rows, self.cols, 0)
        else:
            self.ishape = (0,0,0,0)

    def loadDmsFromBin(self, folder):
        self.ishape = np.load(os.path.join(folder, 'ishape.npy'))
        self.dms = np.load(os.path.join(folder, 'dms.npy'))
        self.skeletonID = np.load(os.path.join(folder, 'skeletonID.npy'))
        self.ishape[3] = np.max(self.dms)

    def saveDms(self, folder, sparse = 10):
        sparse_nums = self.frames / sparse
        s_ishape = (sparse_nums, self.rows, self.cols, 0)
        s_dms = np.zeros((sparse_nums, self.rows, self.cols))
        s_skeletonID = np.zeros((sparse_nums, self.rows, self.cols))

        start_i = (self.frames - sparse_nums*sparse)/2
        for f in xrange(sparse_nums):
            s_dms[f] = self.dms[start_i + f*sparse]
            s_skeletonID[f] = self.skeletonID[start_i + f*sparse]
        np.save(os.path.join(folder, 'ishape.npy'), s_ishape)
        np.save(os.path.join(folder, 'dms.npy'), s_dms)
        np.save(os.path.join(folder, 'skeletonID.npy'), s_skeletonID)

    def scaleDms(self, nshape):
        board = self.globalBoard()
        
        newdms = np.zeros((self.frames, nshape[0], nshape[1]))
        for i in xrange(self.frames):
            newdms[i] = scipy.misc.imresize(self.dms[i][board[0]:board[1], board[2]:board[3]], nshape)
        return newdms

    def rawDMMs(self, nshape):
        board = self.globalBoard()
        rawdmms = np.zeros((self.frames-1, nshape[0], nshape[1]))
        for i in xrange(self.dms.shape[0]-1):
            temp = np.abs(self.dms[i+1] - self.dms[i])
            rawdmms[i] = scipy.misc.imresize(temp[board[0]:board[1], board[2]:board[3]], nshape)
        return rawdmms

    def maps2d(self, mat_array):
        """
        ishape[0]: nFrames
        ishape[1]: Height (rows)
        ishape[2]: Width (cols)
        """
        f2d = np.zeros((self.ishape[1],self.ishape[2]))
        t2d = np.zeros((self.ishape[3],self.ishape[2]))
        s2d = np.zeros((self.ishape[1],self.ishape[3]))

        mat_array.astype(int)
        for r in xrange(self.ishape[1]):
            for c in xrange(self.ishape[2]):
                d = mat_array[r][c]
                if d > 0:
                    f2d[r][c] = 1
                    t2d[d-1][c] = 1
                    s2d[r][d-1] = 1
        return (f2d, t2d, s2d)

    def globalBoard(self):
        total = np.zeros((self.ishape[1], self.ishape[2]))
        for i in xrange(self.dms.shape[0]):
            total += self.dms[i]
        boards = np.nonzero(total)
        board = (np.min(boards[0]), np.max(boards[0])+1, np.min(boards[1]), np.max(boards[1])+1)
        return board

    def DMM(self, pyramid = 0, isize = (15, 10)):
        maps2d = self.maps2d(self.dms[0])
        dmmf = np.zeros((self.dms.shape[0]-1, maps2d[0].shape[0], maps2d[0].shape[1]))
        dmmt = np.zeros((self.dms.shape[0]-1, maps2d[1].shape[0], maps2d[1].shape[1]))
        dmms = np.zeros((self.dms.shape[0]-1, maps2d[2].shape[0], maps2d[2].shape[1]))
        
        idmmf = np.zeros((self.dms.shape[0]-1, isize[0], isize[1]))
        idmmt = np.zeros((self.dms.shape[0]-1, isize[0], isize[1]))
        idmms = np.zeros((self.dms.shape[0]-1, isize[0], isize[1]))

        for i in xrange(self.dms.shape[0]-1):
            maps2d = self.maps2d(self.dms[i])
            maps2d_next = self.maps2d(self.dms[i+1])
            dmmf[i] = np.abs(maps2d[0]-maps2d_next[0])
            dmmt[i] = np.abs(maps2d[1]-maps2d_next[1])
            dmms[i] = np.abs(maps2d[2]-maps2d_next[2])
        
            board = (np.nonzero(dmmf[i]), np.nonzero(dmmt[i]), np.nonzero(dmms[i]))
            idmmf[i] = scipy.misc.imresize(dmmf[i, np.min(board[0][0]):np.max(board[0][0])+1, np.min(board[0][1]):np.max(board[0][1])+1], isize)
            idmmt[i] = scipy.misc.imresize(dmmt[i, np.min(board[1][0]):np.max(board[1][0])+1, np.min(board[1][1]):np.max(board[1][1])+1], isize)
            idmms[i] = scipy.misc.imresize(dmms[i, np.min(board[2][0]):np.max(board[2][0])+1, np.min(board[2][1]):np.max(board[2][1])+1], isize)

            idmmf[i] = idmmf[i] * 189 // np.max(idmmf[i])
            idmmt[i] = idmmt[i] * 189 // np.max(idmmt[i])
            idmms[i] = idmms[i] * 189 // np.max(idmms[i])

        return np.array([idmmf, idmmt, idmms]).astype(int)

    def DMMPyramid(self, pyramid = 0, isize = (240, 160)):
        maps2d = self.maps2d(self.dms[0])
        dmmf = np.zeros(maps2d[0].shape)
        dmmt = np.zeros(maps2d[1].shape)
        dmms = np.zeros(maps2d[2].shape)
        dmmf0 = None
        dmmt0 = None
        dmms0 = None
        dmmf1 = None
        dmmt1 = None
        dmms1 = None
        
        for i in xrange(self.dms.shape[0]-1):
            maps2d_next = self.maps2d(self.dms[i+1])
            dmmf += np.abs(maps2d[0]-maps2d_next[0])
            dmmt += np.abs(maps2d[1]-maps2d_next[1])
            dmms += np.abs(maps2d[2]-maps2d_next[2])
            if i == (1*self.dms.shape[0]/4-1):
                dmmf0 = np.copy(dmmf)
                dmmt0 = np.copy(dmmt)
                dmms0 = np.copy(dmms)
                
        dmmf1 = dmmf - dmmf0
        dmmt1 = dmmt - dmmt0
        dmms1 = dmms - dmms0
        
        board = (np.nonzero(dmmf), np.nonzero(dmmt), np.nonzero(dmms),
                 np.nonzero(dmmf0), np.nonzero(dmmt0), np.nonzero(dmms0),
                 np.nonzero(dmmf1), np.nonzero(dmmt1), np.nonzero(dmms1))
        dmmf = scipy.misc.imresize(dmmf[np.min(board[0][0]):np.max(board[0][0])+1,np.min(board[0][1]):np.max(board[0][1])+1], isize)
        dmmt = scipy.misc.imresize(dmmt[np.min(board[1][0]):np.max(board[1][0])+1,np.min(board[1][1]):np.max(board[1][1])+1], isize)
        dmms = scipy.misc.imresize(dmms[np.min(board[2][0]):np.max(board[2][0])+1,np.min(board[2][1]):np.max(board[2][1])+1], isize)
        dmmf0 = scipy.misc.imresize(dmmf0[np.min(board[3][0]):np.max(board[3][0])+1,np.min(board[3][1]):np.max(board[3][1])+1], isize)
        dmmt0 = scipy.misc.imresize(dmmt0[np.min(board[4][0]):np.max(board[4][0])+1,np.min(board[4][1]):np.max(board[4][1])+1], isize)
        dmms0 = scipy.misc.imresize(dmms0[np.min(board[5][0]):np.max(board[5][0])+1,np.min(board[5][1]):np.max(board[5][1])+1], isize)
        dmmf1 = scipy.misc.imresize(dmmf1[np.min(board[6][0]):np.max(board[6][0])+1,np.min(board[6][1]):np.max(board[6][1])+1], isize)
        dmmt1 = scipy.misc.imresize(dmmt1[np.min(board[7][0]):np.max(board[7][0])+1,np.min(board[7][1]):np.max(board[7][1])+1], isize)
        dmms1 = scipy.misc.imresize(dmms1[np.min(board[8][0]):np.max(board[8][0])+1,np.min(board[8][1]):np.max(board[8][1])+1], isize)

        plt.imshow(dmmf0*3)
        plt.show()
        plt.imshow(dmmf1*3)
        plt.show()
        plt.imshow(dmmf*3)
        plt.show()
        
        return np.array([dmmf, dmmt, dmms,
                         dmmf0, dmmt0, dmms0,
                         dmmf1, dmmt1, dmms1])

    def pureDMMPyramid(self, pyramid = 0, isize = (100, 100)):
        dmmf = np.zeros((self.ishape[1], self.ishape[2]))
        dmmfs = []
        stop_point = set([(i*self.frames/pyramid)-1 for i in range(1, pyramid)])
        prev = np.zeros((self.ishape[1], self.ishape[2]))
        
        for i in xrange(self.dms.shape[0]-1):
            dmmf += np.abs(self.dms[i+1]-self.dms[i])
            if i in stop_point:
                dmmfs.append(np.abs(dmmf-prev))
                prev = np.copy(dmmf)
        dmmfs.append(np.abs(dmmf-prev))
              
        board = self.globalBoard()
        pureout = np.zeros((pyramid, isize[0], isize[1]))
        for i in xrange(pyramid):
            pureout[i] = scipy.misc.imresize(dmmfs[i][board[0]:board[1], board[2]:board[3]], isize)

        return pureout

            
# if __name__ == '__main__':
#     io = DepthMapBinFileIO()
#     # io.loadDms("F:/data/S1/a00_s01_e01_depth.bin")
#     # print io.frames, io.cols, io.rows
#     # for i in xrange(10):
#     #     plt.imshow(io.dms[i*30])
#     #     plt.show()
#     #     plt.imshow(io.skeletonID[i*30])
#     #     plt.show()

#     io.loadDmsFromBin("F:\\data\\dms\\a00_s01_e00")
#     io.DMM()

    #io.pureDMMPyramid(pyramid = 4)
    #plt.imshow(total)
    #plt.show()



