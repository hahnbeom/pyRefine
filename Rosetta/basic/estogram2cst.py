## this is estogram2cst in pyRosetta version

import sys,os,copy
import sys
import rosetta_utils
import numpy as np
import pyrosetta as PR

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = SCRIPTDIR+'/../data/'

DELTA = np.array([-25.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.75,
                  0.0,
                  0.75,1.5,3.0,7.0,12.5,17.5,25.0])
NBINS = len(DELTA)
kCEN = int((len(DELTA)-1)/2)

##############################################
# CONSTANTS (can be moved to opt class if necessary
DCUT_SOFT = 35.0 # Max input model's (i,j) distance for spline restraints
DCUT_HARD = 20.0 # Max input model's (i,j) distance for hard restraints
MIND = 4.0  # Minimum distance in spline x
DCAP = 40.0 # Maximum distance in spline x

PCORE_DFLT=[0.6,0.7,0.8] #at what Pcen to apply hard cst? (will skip otherwise)
TOL_DFLT  =[4.0,2.0,1.0] #Tolerance of flat-harmonic function
SIGMA_DFLT=[2.5,2.0,1.0] #Sigma of flat-harmonic function

# goes to "opt"
#FUNC = ['bounded','sigmoid'] #cen/fa for Pcen > PCORE[1]
#PCON_BOUNDED = 0.9
#P_spline_on_fa = 0.6

class default_opt:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

#####################################################
class Pair:
    def __init__(self,estogram,id1,id2,d0):
        self.ids = (id1,id2) #Rosetta AtomID
        self.estogram = estogram
        self.Pcorr = estogram #placeholder for uncorrected
        self.d0 = d0
        self.Pcen = sum(estogram[kCEN-1:kCEN+2])
        self.seqsep = abs(id1.rsd()-id2.rsd())

    def P_dfire(self): #no reference prob
        Pcorr = copy.deepcopy(self.estogram)
        Pref = self.estogram[kCEN] #single value at center
        for i,dd in enumerate(DELTA):
            d = max(5.0,self.d0-dd)
            gamma = (d/self.d0)**1.61
            Pcorr[i] /= gamma*Pref
        return Pcorr

    def do_correction_dfire(self,Pref=[]):
        Pcorr = self.P_dfire()
        if Pref != []:
            Pcorr /= Pref
        self.Pcorr = Pcorr/np.sum(Pcorr)
        
    def do_correction_stat(self,Pref):
        Pcorr = self.estogram/Pref
        self.Pcorr = Pcorr/np.sum(Pcorr)

    def calc_Pcontact(self):
       self.Pcontact = 0.0
       for k,P in enumerate(self.Pcorr):
            if   self.d0 - DELTA[k] < 8.0 : self.Pcontact += P
            elif self.d0 - DELTA[k] < 10.0: self.Pcontact += 0.5*P
        
    # all the func other than spline
    def estogram2core(self,functype,
                      Pcore=PCORE_DFLT,
                      Sigma=SIGMA_DFLT,
                      Tol  =TOL_DFLT,
                      out=None):

        k = 0 
        if self.Pcen > Pcore[2]: k = 2
        elif self.Pcen > Pcore[1]: k = 1
        # otherwise 0

        if k == 0 and self.seqsep <= 12: return []

        tol = Tol[k]
        sig = Sigma[k]
        
        funcs = []
        if functype == 'bounded':
            funcs.append(rosetta.core.scoring.constraints.BoundFunc(self.d0-tol,self.d0+tol,
                                                                    sig,""))
        elif functype == 'fharm':
            funcs.append(rosetta.core.scoring.func.FlatHarmonicFunc(self.d0,sig,tol))
        elif functype == 'sigmoid':
            w = 1.0
            m = 5.0/sig
            x1 = d0-tol
            x2 = d0+tol
            func1 = (-w,d0-tol,m,w)
            func2 = ( w,d0+tol,m,w)
            funcs.append(func1)
            funcs.append(func2)
            
        if out != None:
            i = self.ids[0].rsd()
            j = self.ids[1].rsd()
            form = '%3d %3d %8.3f %-7s'+' Tol/Sig/Pcore: %6.3f %6.3f %6.3f\n'
            out.write(form%tuple([i,j,self.d0,functype.upper()]+[tol,sig,self.Pcen]))
        return funcs

    def estogram2spline(self,maxP_spline_on=0.0,out=None):
        maxP = max(self.Pcorr)
        if maxP <= maxP_spline_on: return False

        xs = self.d0+DELTA
        ys = -np.log(self.Pcorr+1.0e-4)
        ys = np.flip(ys,0) # make as ascending order in distance
        
        ys -= ys[7] #subtract center
        
        xs_v = PR.rosetta.utility.vector1_double()
        ys_v = PR.rosetta.utility.vector1_double()
        for k,x in enumerate(xs):
            if x < 1.0:
                xs[k] = 1.0
                ys[k] = 9.999
                continue
            xs_v.append(x)
            ys_v.append(ys[k])
        ys_v[1] = max(2.0,ys[2]+2.0) # soft repulsion
            
        func = PR.rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, 1.0, xs_v, ys_v )
        if out != None:
            i = self.ids[0].rsd()
            j = self.ids[1].rsd()
            form = '%3d %3d %8.3f SPLINE '+' %6.3f'*len(ys)+'\n'
            out.write(form%tuple([i,j,self.d0]+list(ys)))
        
        return func

    def estogram2contact(self,out=None):
        if out != None:
            i = self.ids[0].rsd()
            j = self.ids[1].rsd()
            form = '%3d %3d %8.3f %-7s'+' Tol/Sig/Pcont: %6.3f %6.3f %6.3f\n'
            out.write(form%tuple([i,j,self.d0,"BOUNDED"]+[4.0,1.0,self.Pcontact]))
        return PR.rosetta.core.scoring.constraints.BoundFunc(4.0,12.0,1.0,"") #still in constraints...
    
#####################################################
#### Reference state 
# list of conditioners
def lookup_d(Pref,p):
    dbin = int(p.d0-4.0)
    if dbin < 0: dbin = 0
    elif dbin >= max(Pref.keys()): dbin = max(Pref.keys())
    return Pref[dbin]

def lookup_d_sep(Pref,p):
    Pref_d = lookup_d(Pref,p)
    seqsep = min(max(Pref_d.keys()),p.seqsep)
    return Pref_d[seqsep]

def lookup_d_sep_Q(Pref,p):
    Pref_d_sep = lookup_d_sep(Pref,p)
    Qbin = min(5,max(0,int(p.Q-0.4)/0.1))
    return Pref_d_sep[Qbin]
#############

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

def do_reference_correction(pairs,estogram,Q,opt='statQ'):
    # load list of reference states
    # read_Pref returns Pref & conditioner()
    if opt[:4] == 'stat':
        Pref,conditioner = read_Pref(DATAPATH+'/refstat.dbin.seqsep.txt')
        for p in pairs:
            Pref_at_p_condition = conditioner(Pref,p)
            p.do_correction_stat(Pref_at_p_condition)

    elif opt[:5] == 'dfire':
        if opt == 'dfire':
            p.do_correction_dfire()
        elif opt == 'dfireQ':
            Qref = read_Qref(DATAPATH+'/refstat.Q',Q)
            p.do_correction_dfire(Pref=Qref)

    #else: pass ##no correction
    
    # Get Prob. contacting
    # should be done after reference correction
    for p in pairs:
        p.calc_Pcontact() 
            
#####################################################
            
def find_pairs(pose,estogram,dcut=35.0):
    nres = len(estogram)

    pairs = []
    for i in range(nres-4):
        if pose.residue(i+1).has('CB'): atm_i = 'CB'
        else: atm_i = 'CA'
        atm_i = pose.residue(i+1).atom_index(atm_i)
        id1 = PR.rosetta.core.id.AtomID(atm_i,i+1)
        
        for j in range(i+4,nres):
            if pose.residue(i+1).has('CB'): atm_j = 'CB'
            else: atm_j = 'CA'
            atm_j = pose.residue(i+1).atom_index(atm_j)
            id2 = PR.rosetta.core.id.AtomID(atm_j,j+1)

            xyz_i = pose.xyz(id1)
            xyz_j = pose.xyz(id2)
            d0 = xyz_i.distance(xyz_j)
            
            if d0 > dcut: continue

            p = Pair(estogram[i][j],id1,id2,d0)
            pairs.append(p)
    return pairs

# To regularize super-hard cases
def dynamic_Pcore_cut( pairs,cut0,ncore_cut ):
    cut = cut0
    while True:
        ncore = 0
        for p in pairs:
            (id1,id2) = p.ids
            if p.seqsep < 12: continue #skip short seqsep
            if p.Pcen >= cut:
                ncore += 1
        if ncore >= ncore_cut: break
        cut -= 0.05
        
    return cut

def apply_on_pose( pose, npz, opt, debug=False, reportf=None ):
    dat = np.load(npz)
    estogram = dat['estogram']    
    Q = np.mean(dat['lddt'])

    # 1. get list of valid pairs within dcut
    pairs = find_pairs(pose,estogram,dcut=DCUT_SOFT)

    # 2. apply reference correction
    do_reference_correction(pairs,estogram,Q,
                            opt=opt.refcorrection_method)

    out = None
    if reportf != None: out = open(reportf,'w')

    # 3. Get lowest-Pcore cut for hard cases (in normal case ==opt.Pcore[0])
    nres = len(estogram)
    ncore_cut = float(nres)*np.log(float(nres))

    Pcore_cut = dynamic_Pcore_cut( pairs, cut0=opt.Pcore[1], ncore_cut=ncore_cut )
    print( "Decide Pcore_0 cut as %5.2f"%Pcore_cut )

    # 4. make list of csts
    ncore,nspl,ncont = (0,0,0)
    for p in pairs:
        (id1,id2) = p.ids

        funcs = []
        if p.d0 <= DCUT_SOFT:
            funcs.append( p.estogram2spline(maxP_spline_on=opt.Pspline_on_fa,
                                            out=out) )
            nspl += 1

            if p.Pcontact > opt.Pcontact_cut and p.Pcen < opt.Pcore[1]:
                funcs.append( p.estogram2contact(out=out) )
                ncont += 1

            
        if p.d0 <= DCUT_HARD and p.Pcen >= Pcore_cut:
            funcs += p.estogram2core(functype=opt.hardcsttype,
                                     out=out)
            ncore += 1
            
        for func in funcs:
            if not func: continue
            cst = PR.rosetta.core.scoring.constraints.AtomPairConstraint( id1, id2, func )
            pose.add_constraint( cst )

    if reportf != None: out.close()
    print( "Applied %d spline / %d core / %d contact restraints."%(nspl,ncore,ncont) )
    
    # just for debugging...
    if debug:
        sfxn = PR.create_score_function("empty")
        sfxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 1.0)
        E = sfxn.score(pose)
        print( E )
    
if __name__ == "__main__":
    PR.init('-mute all')

    pdb = sys.argv[1]
    npz = sys.argv[2]
    pose = PR.pose_from_file(pdb)
    
    opt = default_opt( Pcore=[0.6,0.7,0.8],
                       Pspline_on_fa=0.0, #cen
                       Pcontact_cut=0.9,
                       hardcsttype="bounded",
                       refcorrection_method="statQ")
    
    apply_on_pose( pose, npz, opt, debug=True, reportf="cst.txt" )
