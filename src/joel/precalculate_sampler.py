import numpy as np
import pickle
import sys
import os.path

hereDir = os.path.dirname(os.path.abspath(__file__))

sys.path.extend([os.path.join(hereDir, '../cnn'),
                 os.path.join(hereDir, '..')])
from constants import *
from brainroller.shtransform import SHTransformer

out_basename = 'precalc_sampler'

edge_len = 2 * RAD_PIXELS + 1
r_max = 0.5*float(edge_len + 1)

low_plane = int(sys.argv[1])
assert low_plane >= 0 and low_plane <= edge_len - 1, 'bad low plane'
high_plane = int(sys.argv[2])
assert high_plane > low_plane and high_plane <= edge_len, 'bad high plane'

transformer = SHTransformer(edge_len, MAX_L)
indexL = []
valL = []
dense_shape = None
for in_idx_x in range(low_plane, high_plane):
    indexSubL = []
    valSubL = []
    for in_idx_y in range(edge_len):
        for in_idx_z in range(edge_len):
            #print('in coords: ', in_idx_x, in_idx_y, in_idx_z)
            in_mtx = np.zeros([edge_len, edge_len, edge_len])
            in_mtx[in_idx_x, in_idx_y, in_idx_z] = 1.0
            in_flat_idx = np.nonzero(in_mtx.flat)
            #print('flat: ', in_flat_idx)
            in_flat_idx = in_flat_idx[0][0]
            rslt = transformer.calcBallOfSamples(in_mtx)
            if dense_shape is None:
                dense_shape = [edge_len*edge_len*edge_len, rslt.shape[0]]
                print('dense_shape: ', dense_shape)
            nz_indices = np.nonzero(rslt)
            #print('%d %d %d -> %s entries -> %s' % (in_idx_x, in_idx_y, in_idx_z, nz_indices, rslt[nz_indices]))
            for out_idx, val in zip(nz_indices[0], rslt[nz_indices]):
                #print('in_flat_idx: ', in_flat_idx, ' out_idx: ', out_idx, ' val: ', val)
                indexSubL.append([in_flat_idx, out_idx])
                valSubL.append(val)
    print('%d %d %d: %d entries' % (in_idx_x, in_idx_y, in_idx_z, len(valSubL)))
    indexM = np.array(indexSubL, dtype=np.int32)
    valM = np.array(valSubL)
    indexM.tofile("%s_index_%d_%d.npz" % (out_basename, low_plane, high_plane))
    valM.tofile("%s_values_%d_%d.npz" % (out_basename, low_plane, high_plane))
#     indexL.append(np.array(indexSubL))
#     valL.append(np.array(valSubL))
# with open('%s.pkl' % out_basename, 'wb') as f:
#     pickle.dump([indexL, valL, dense_shape], f)
# print('Wrote %s' % out_fname)
 