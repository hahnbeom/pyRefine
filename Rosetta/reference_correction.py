import sys,os,glob,copy
import numpy as np
import multiprocessing as mp

MYFILE = os.path.abspath(__file__)
DATAPATH = MYFILE.replace(MYFILE.split('/')[-1],'')+'../data'

NBINS = 15
EBINS = [-20.0,-15.0,-10.0,-4.0,-2.0,-1.0,-0.5,
         0.5,1.0,2.0,4.0,10.0,15.0,20.0,
         99] #bining positions: 14

DELTA = [-25.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.75,
         0.0,
         0.75,1.5,3.0,7.0,12.5,17.5,25.0] #central positions in bin: 15

EXCLUDE_NULL = [False for e in DELTA]

# evaluate through GDT-TS style using multiple thresholds
#XSBINS    = [-np.log(0.8),-np.log(0.6),-np.log(0.4),-np.log(0.2)] # Xentropy bins: (0.223,0.511,0.916,1.61)
XSBINS    = [1.6,2.0,2.71] #1.6:uni-across-third 2.0:uni-across-half 2.71:uniform
DDEXPBINS = [1.0,2.0,4.0] #abs(delta) of dexp
GAPBINS   = [1.0/2.71,1.0,2.71] #P_ratio goes to energy...: within +1,0,-1 kcal/mol

#####################################################
class Pair:
    def __init__(self,estogram,i,j,d0,Q):
        self.res = (i,j)
        self.estogram = estogram
        self.d0 = d0
        self.Q = Q
        
    def P_dfire(self): #no reference prob
        Pcorr = copy.deepcopy(self.estogram)
        Pref = self.estogram[7] #single value at center
        for i,dd in enumerate(DELTA):
            d = max(3.0,self.d0-dd)
            gamma = (d/self.d0)**1.61
            Pcorr[i] /= gamma*Pref
        return Pcorr

    def do_correction_dfire(self,Pref=[]):
        Pcorr = self.P_dfire()
        if Pref != []:
            Pcorr /= Pref
        return Pcorr/np.sum(Pcorr)
        
    def do_correction_normal(self,Pref):
        Pcorr = self.estogram/Pref
        return Pcorr/np.sum(Pcorr)

#####################################################
#list of conditioners
def lookup_d(Pref,p):
    dbin = int(p.d0-4.0)
    if dbin < 0: dbin = 0
    elif dbin >= max(Pref.keys()): dbin = max(Pref.keys())
    return Pref[dbin]

def lookup_d_sep(Pref,p):
    (i,j) = p.res
    Pref_d = lookup_d(Pref,p)
    seqsep = min(max(Pref_d.keys()),abs(i-j))
    return Pref_d[seqsep]

def lookup_d_sep_Q(Pref,p):
    Pref_d_sep = lookup_d_sep(Pref,p)
    Qbin = min(5,max(0,int(p.Q-0.4)/0.1))
    return Pref_d_sep[Qbin]

#####################################################

def read_Pref(txt):
    Pref = {}
    reftype = ''
    MEFF = 1.0e-6
    
    for l in open(txt).readlines():
        words = l[:-1].split()
        if len(words) == len(DELTA)+3: #dbin only
            ibin = int(words[0])
            #d = int(words[1])
            Ps = np.array([float(word)+MEFF for word in words[3:]])
            Pref[ibin] = Ps
            reftype = 'd'
        elif len(words) == len(DELTA)+4: #seqsep
            ibin = int(words[0])
            if ibin not in Pref: Pref[ibin] = {}
            seqsep = int(words[2])
            Ps = np.array([float(word)+MEFF for word in words[4:]])
            Pref[ibin][seqsep] = Ps
            reftype = 'd_sep'
        elif len(words) == len(DELTA)+5: #seqsep & Q
            ibin = int(words[0])
            seqsep = int(words[2])
            Qbin = int(words[3])
            if ibin not in Pref: Pref[ibin] = {}
            if seqsep not in Pref[ibin]: Pref[ibin][seqsep] = {}
            Ps = np.array([float(word)+MEFF for word in words[5:]])
            Pref[ibin][seqsep][Qbin] = Ps
            reftype = 'd_sep_Q'

    conditioner = {'d':lookup_d,
                   'd_sep':lookup_d_sep,
                   'd_sep_Q':lookup_d_sep_Q
                   }[reftype]
    return Pref,conditioner
        
def find_pairs(dat,dmtrx,dcut=20.0):
    Q = np.mean(dat['lddt'])
    estogram = dat['estogram']
    
    nres = len(estogram)
    pairs = []
    for i in range(nres-9):
        for j in range(i+9,nres):
            d0 = dmtrx[i][j]
            if d0 > dcut:
                continue
            p = Pair(estogram[i][j],i,j,d0,Q)
            pairs.append(p)
    return pairs

def read_Qref(infile,Q):
    Qref = []
    for l in open(infile):
        words = l[:-1].split()
        Qref.append(np.array([float(word) for word in words[1:]]))

    fQ = (Q-0.4)/0.1
    if fQ >= 5.0:   return Qref[-1]
    elif fQ <= 0.0: return Qref[0]
    else:
        i1 = int(fQ)
        i2 = i1+1
        f = (fQ-i1)
        return (1.0-f)*Qref[i1] + f*Qref[i2] #linear interpolation
    
def run(dmtrx,npz,opt,dcut=30.0):
    dat = np.load(npz)
    n = len(dat['estogram'])
    Q = np.mean(dat['lddt'])

    # stack contacting ones only
    pairs = find_pairs(dat,dmtrx,dcut)
    
    # load list of reference states
    # read_Pref returns Pref & conditioner()
    if opt == 'd_sep':
        Pref = read_Pref(DATAPATH+'/refstat.dbin.seqsep.txt')
    elif opt == 'd_sep_Q':
        Pref = read_Pref(DATAPATH+'/refstat.dbin.seqsep.Q.txt')
    elif opt == 'dfireQ':
        Qref = read_Qref(DATAPATH+'/refstat.Q',Q)
    elif opt == 'none':
        return dat['estogram']

    Pcorr = dat['estogram'] #np.zeros((n,n,NBINS))
    # evaluate
    for p in pairs:
        i,j = p.res
        
        if opt == 'dfire':
            P = p.do_correction_dfire()
        elif opt == 'dfireQ':
            P = p.do_correction_dfire(Pref=Qref)
        elif opt != 'none': 
            Pref,conditioner = Prefs[opt]
            Pref_at_p_condition = conditioner(Pref,p)
            P = p.do_correction_normal(Pref_at_p_condition,opt)#,report=debug)
        Pcorr[i][j] = Pcorr[j][i] = P
        
    return Pcorr
