import sys,os
import numpy as np
import tensorflow as tf

BINS = np.arange(-170, 180, 10)
COLUMNS = ['pdb', 'chain', 'seqpos', 'AA', 'SS', 'Phi', 'Psi', 'Omega', 'CA_x', 'CA_y', 'CA_z']
TOR_COL = ['Phi', 'Psi', 'Omega']
FMT = " %4s %s %5d %s %s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
eps = 1e-9
TOR_CUTOFF = 10.0
MAX_IDX = 1000

N_AA = 20 # regular aa
N_AA_MSA = 21 # regular aa + gap
WMIN = 0.8

w_SEQ = 1.0
w_STR = 4.0
w_SS  = 5.0
w_PHI = 1.0
w_PSI = 0.8
w_OMG = 0.8

def parse_opts(argv):
    import argparse
    opt = argparse.ArgumentParser\
            (description='''FragPicker: Pick fragment from vall database based on predicted torsion angles''')
    #
    opt.add_argument('-title', dest='title', metavar="TITLE", required=True,\
                         help="Prefix for output file. output file will be title.?mers")
    opt.add_argument('-a3m_fn', dest='a3m_fn', metavar="a3m_fn", required=True,\
                         help="MSA file in a3m format")
    opt.add_argument('-tor_pred', dest='pred_tor_fn', metavar="TOR_PRED", required=True,\
                         help="Predicted torsion angle file from ML model.")
    opt.add_argument('-SS_pred', dest='pred_SS_fn', metavar="SS_PRED", required=True,\
                         help="Predicted SS file from ML model.")
    opt.add_argument('-device_id', '--device_id', dest='device_id', metavar='N_device_id', nargs='+', type=int, default=[0], \
                         help='Which gpu to be used [0]')
    opt.add_argument('-n_frag', dest='n_frag', metavar='N_FRAGS', type=int, default=200, \
                         help="The number of fragments per position [200]")
    opt.add_argument('-n_mer', dest='n_mer', metavar='N_MER', type=int, nargs='+',
                     help='Length of fragment [9]')
    opt.add_argument('-batch_size', dest='batch_size', metavar='BATCH', type=int, default=64, \
                         help='Batch size for gpu calculation [64]')
    opt.add_argument('-vall_full', dest='vall_full', metavar='VALL_FULL', \
                         default='%s/data/vall.jul19.2011.json'%vall_home,\
                         help="json file containing all necessary information of vall database")
    opt.add_argument('-vall_fn', dest='vall_fn', metavar='VALL', \
                         default='%s/data/vall.jul19.2011.vonMises.npy'%vall_home,\
                         help="one-hot encoded BB torsion angles in vall")
    
    if len(argv) == 1:
        opt.print_help()
        return
    params = opt.parse_args()
    return params

def read_msa(a3m_fn):
    import string

    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file
    with open(a3m_fn) as fp:
        for line in fp:
            if line[0] == '>': continue
            seqs.append(line.rstrip().translate(table)) # remove lowercase letters
    # convert letters into numbers
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    msa[msa > 20] = 20
    return seqs[0], msa

def reweight_seq(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff # msa1hot.shape[1] == n_res
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot[:,:,:20], axis=0) / beff + 1e-9

    return f_i

class Picker(object):
    def __init__(self, sess, title, n_frag=50, n_mer=9, batch_size=64,
                 vall_fn="/home/minkbaek/DeepLearn/torsion/vall/data/vall.jul19.2011.vonMises.npy",
                 vall_SS_fn="/home/minkbaek/DeepLearn/torsion/vall/data/vall.jul19.2011.SS3.npy",
                 vall_seq_fn="/home/minkbaek/DeepLearn/torsion/vall/data/vall.jul19.2011.SeqProf.npy",
                 vall_str_fn="/home/minkbaek/DeepLearn/torsion/vall/data/vall.jul19.2011.StrProf.npy",
                 vall_full_fn="/home/minkbaek/DeepLearn/torsion/vall/data/vall.jul19.2011.json"):
        self.sess = sess
        self.title = title
        self.n_frag = n_frag
        self.n_mer = n_mer
        self.batch_size = batch_size
        self.n_ang = 36+36+2
        
        sys.stdout.write("INFO: Read vall files\n")
        self.vall_SStor, self.vall_seq, self.vall_str, self.vall_full = self.load_data(vall_fn, vall_SS_fn, vall_seq_fn, vall_str_fn, vall_full_fn)
        sys.stdout.write("INFO: Prepare tensorflow model\n")
        self.build_model()
        
    def load_data(self, vall_fn, vall_SS_fn, vall_seq_fn, vall_str_fn, vall_full_fn):
        import pandas as pd
        vall_full = pd.read_json(vall_full_fn, orient='records')
        #
        vall = np.load(vall_fn)
        vall[np.where(vall < -50)[0]] = 100 
        vall_SS = np.load(vall_SS_fn)
        vall_seq = np.load(vall_seq_fn)
        vall_str = np.load(vall_str_fn)

        vall = vall.astype(np.float32)
        vall_SS = vall_SS.astype(np.float32)
        

        return np.concatenate((vall, vall_SS), axis=-1), vall_seq.astype(np.float32), vall_str.astype(np.float32), vall_full
        
    def build_model(self):
        # input
        self.db_SStor = tf.placeholder(tf.float32, [None, self.n_ang+3])
        self.db_seq = tf.placeholder(tf.float32, [None, 20])
        self.db_str = tf.placeholder(tf.float32, [None, N_AA])
        self.SStor = tf.placeholder(tf.float32, [None, self.n_ang+3])
        self.msa = tf.placeholder(tf.uint8, [None, None]) # n_seq, n_res
        #
        # get pssm features from MSA
        n_res = tf.shape(self.msa)[-1]
        msa1hot = tf.one_hot(self.msa, N_AA_MSA, dtype=tf.float32)
        w_seq = reweight_seq(msa1hot, WMIN)
        q_prof = msa2pssm(msa1hot, w_seq)
        del msa1hot, w_seq
        #
        # calculate profile difference (using tf.while_loop)
        prof_raw = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def calc_prof_diff(i, prof_raw):
            bias = q_prof[i,:]
            diff = tf.nn.bias_add(self.db_seq, bias)
            diff = tf.reduce_sum(tf.abs(diff), axis=-1)
            prof_raw = prof_raw.write(i, diff)
            return i+1, prof_raw
        i = 0
        cond = lambda i, prof_raw: tf.less(i, n_res)
        _, prof_raw = tf.while_loop(cond, calc_prof_diff, [i, prof_raw])
        prof_raw = prof_raw.stack()
        #
        prof_raw = tf.reshape(prof_raw, [1, n_res, -1, 1])
        filters = tf.eye(self.n_mer)
        filters = tf.reshape(filters, [self.n_mer, self.n_mer, 1, 1])
        score_seq = tf.nn.conv2d(input=prof_raw, filter=filters, strides=[1,1,1,1], padding='VALID')
        score_seq = tf.squeeze(score_seq)
        del prof_raw
        #
        # calculate str profile difference (using tf.while_loop)
        prof_raw = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def calc_StrProf_diff(i, prof_raw):
            bias = q_prof[i,:]
            diff = tf.nn.bias_add(self.db_str, bias)
            diff = tf.reduce_sum(tf.abs(diff), axis=-1)
            prof_raw = prof_raw.write(i, diff)
            return i+1, prof_raw
        i = 0
        cond = lambda i, prof_raw: tf.less(i, n_res)
        _, prof_raw = tf.while_loop(cond, calc_StrProf_diff, [i, prof_raw])
        prof_raw = prof_raw.stack()
        #
        prof_raw = tf.reshape(prof_raw, [1, n_res, -1, 1])
        score_str = tf.nn.conv2d(input=prof_raw, filter=filters, strides=[1,1,1,1], padding='VALID')
        score_str = tf.squeeze(score_str)
        del prof_raw, q_prof
        #
        # calculate SStor similarity (using tf.while_loop)
        SStor = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def calc_SStor_sim(i, SStor):
            prob = self.SStor[i,:]
            sim = tf.reduce_sum(tf.math.multiply(self.db_SStor, prob[None,:]), axis=-1)
            SStor = SStor.write(i, sim)
            return i+1, SStor
        i = 0
        cond = lambda i, SStor: tf.less(i, n_res)
        _, SStor = tf.while_loop(cond, calc_SStor_sim, [i, SStor])
        SStor = SStor.stack()
        SStor = tf.reshape(SStor, [1, n_res, -1, 1])
        score_SStor = tf.nn.conv2d(input=SStor, filter=filters, strides=[1,1,1,1], padding='VALID')
        score_SStor = tf.squeeze(score_SStor)
        del SStor
        #
        self.score = w_SEQ * score_seq + w_STR * score_str + score_SStor
        self.idx = tf.argsort(self.score, axis=-1)
        
    def calc_score(self, a3m_fn, pred, SS_raw):
        sys.stdout.write("INFO: Calculate fragment scores for %d-mer\n"%self.n_mer)
        # read msa
        seq, msa = read_msa(a3m_fn)
        #
        # read SS prob
        #SS_raw = np.load(SS_fn)
        SS = np.zeros((len(seq), 3), dtype=np.float32) # 3-state SS
        SS[:,0] = np.sum(SS_raw[:, :2], axis=-1) # state = E
        SS[:,1] = np.sum(SS_raw[:,2:5], axis=-1) # state = H
        SS[:,2] = np.sum(SS_raw[:,5: ], axis=-1) # state = C
        SS[0,:] = 1.0
        SS[-1,:] = 1.0
        SS = w_SS * (1.0-SS)
        #
        # read tor prob
        #pred = np.load(tor_fn)
        pred = pred.astype(np.float32)
        pred[0,:36] = 1.0 # mask 1st residue's phi ang prediction
        pred[-1,36:] = 1.0 # mask last residue's psi/omega ang prediction
        #
        pred[:,  :36] = w_PHI * (1.0 - pred[:,  :36])
        pred[:,36:72] = w_PSI * (1.0 - pred[:,36:72])
        pred[:,72:  ] = w_OMG * (1.0 - pred[:,72:  ])
        #
        pred = np.concatenate((pred, SS), axis=-1)
        #
        vall_total = self.vall_seq.shape[0]
        n_batch = 4 # hard coded now.. but it should be changed someday
        if vall_total%n_batch > self.n_mer-1:
            n_batch += 1
        n_elem = vall_total // n_batch
        score_s = list()
        idx_s = list()
        for i_batch in range(n_batch):
            start = n_elem*i_batch
            end = min(n_elem*(i_batch+1) + self.n_mer - 1, vall_total)
            score, idx = self.sess.run([self.score, self.idx],\
                    feed_dict={self.db_SStor: self.vall_SStor[start:end, :],
                        self.SStor: pred,
                        self.msa:msa, self.db_seq: self.vall_seq[start:end, :],
                        self.db_str: self.vall_str[start:end, :]})
            score_s.append(np.array(score))
            idx_s.append(np.array(idx))
        score = np.concatenate(score_s, axis=-1)
        idx = np.concatenate(idx_s, axis=-1)
        return score, idx
    
    def pick(self, idx_s):
        sys.stdout.write("INFO: Pick top %d fragments per position\n"%self.n_frag)
        fmt_frag = "\n".join([FMT for i in range(self.n_mer)])
        wrt = []
        for i_res in range(idx_s.shape[0]):
            idx = idx_s[i_res, :]
            wrt.append("position: %12d neighbors: %12d"%(i_res+1, self.n_frag))
            wrt.append('')
            i_frag = 0
            frag_tor_s = np.array(self.vall_full[idx[i_frag]:idx[i_frag]+self.n_mer][TOR_COL])
            frag_tor_s = frag_tor_s[np.newaxis, :, :]
            n_frag = 1
            frags = np.array(self.vall_full[idx[i_frag]:idx[i_frag]+self.n_mer][COLUMNS])
            wrt.append(fmt_frag%tuple(frags.flatten()))
            wrt.append("")
            #
            redun_idx_s = list()
            for i_frag in range(1, MAX_IDX):
                frag_tor = np.array(self.vall_full[idx[i_frag]:idx[i_frag]+self.n_mer][TOR_COL])
                #
                tor_diff = frag_tor_s - frag_tor[None, :, :]
                tor_diff = np.abs((tor_diff + 180.0) % 360.0 - 180.0) < TOR_CUTOFF
                tor_diff = np.all(tor_diff, axis=(1,2))
                if np.any(tor_diff):
                    redun_idx_s.append(idx[i_frag])
                    continue
                frag_tor_s = np.concatenate([frag_tor_s, frag_tor[np.newaxis, :, :]], axis=0)
                #
                frags = np.array(self.vall_full[idx[i_frag]:idx[i_frag]+self.n_mer][COLUMNS])
                wrt.append(fmt_frag%tuple(frags.flatten()))
                wrt.append("")
                #
                n_frag += 1
                if n_frag == self.n_frag:
                    break
            #
            if n_frag < self.n_frag:
                for i_frag in range(self.n_frag - n_frag):
                    frags = np.array(self.vall_full[redun_idx_s[i_frag]:redun_idx_s[i_frag]+self.n_mer][COLUMNS])
                    wrt.append(fmt_frag%tuple(frags.flatten()))
                    wrt.append("")
        #
        with open("%s.%dmers"%(self.title, self.n_mer), 'w') as fp:
            fp.write("\n".join(wrt))

def main(params=None, pred_tor_np=[], pred_SS_np=[], title=None, n_mers=[] ):
    if params == None:
        params = parse_opts(sys.argv[1:])
        
    if len(pred_tor_np) == 0: 
        pred_tor_np = np.load(params.pred_tor_fn)
    if len(pred_SS_np) == 0:
        pred_SS_np = np.load(params.pred_SS_fn)
    if len(n_mers) == 0:
        n_mers = params.n_mer
            
    if len(pred_tor_np) == 0 or len(pred_SS_np) == 0:
        sys.exit('Numpy objects for SS and/or Tors prediction not read properly!')
    if len(n_mers) == 0:
        sys.exit('No n_mer defined!')

    if title == None: title = params.title
    
    run_config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.92))

    with tf.Session(config=run_config) as sess:
        picker = Picker(sess, title, n_frag=params.n_frag,
                        batch_size=params.batch_size,
                        vall_fn=params.vall_fn, vall_full_fn=params.vall_full)
        for n_mer in n_mers:
            picker.n_mer = n_mer
            score, idx = picker.calc_score(params.a3m_fn, pred_tor_np, pred_SS_np)
            picker.pick(idx)

if __name__ == '__main__':
    main()
