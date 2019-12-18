import sys,os
import math
import numpy as np
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/../utils'%SCRIPTPATH)
import myutils

##############################################
def parse_opts(argv):
    import argparse

    opt = argparse.ArgumentParser\
            (description='''Get error prediction and returns Rosetta cst files''')
    # Required
    opt.add_argument('-npz', dest='npz', required=True, help="Error prediction file in npz format")
    opt.add_argument('-pdb', dest='pdb', required=True, help="MSA file in a3m format")

    # optional
    opt.add_argument('-prefix', dest='prefix', help="")
    opt.add_argument('-softcsttype', dest='softcst', default='spline', help="functional form of softcst")
    opt.add_argument('-splinedir', dest='splinedir', help="Where to dump spline files.")

    opt.add_argument('-crdcst', dest='crdcst', default=True, action="store_true", help="")
    opt.add_argument('-ulr_pred', dest='ulr_pred', default=True, action="store_true", help="")
    opt.add_argument('-ulr', dest='ulr_s', nargs='+', required=True, \
                     help='ULRs designated (e.g. 5-10 16-20). Will use predicted if not provided')
    
    if len(argv) == 1:
        opt.print_help()
        return
    
    params = opt.parse_args()
    if params.prefix == None:
        params.prefix = params.pdb[:-4]
    if params.ulr_s != []:
        params.ulrs_static = [range(int(word.split('-')[0]),int(word.split('-')[1])) for word in params.ulr_s]
        
    return params

def calc_Pcore(Ps):
    return sum(Ps[6:9]) #CAUTION -- hardcoded

class PairCstClass:
    def __init__(self,Ps,distinfo,csttype,cst_p,
                 ebin_defs=None):
        self.Ps = Ps
        self.csttype = csttype
        self.distinfo = distinfo #res1,res2,atm1,atm2,d0
        self.cst_p = cst_p
        self.ebin_defs = ebin_defs
        self.m_v = [] #only for SOG func; defined at fitSOG

    def P2spline(self):
        outf = self.cst_p[0]
        d0 = self.distinfo[4]
        kCEN = int(len(self.ebins_defs)/2)
        xs = []
        ys = []
        Pref = self.Ps[kCEN] 
        Ps_count = []
        Ps_count_u = []

        MEFF=0.0001
        for k,P in enumerate(self.Ps):
            d = d0 - self.ebin_defs[k]
            E = -math.log((P+MEFF)/Pref)
            xs.append(d)
            ys.append(E)

        xs.reverse()
        ys.reverse()

        kss = [0]
        ksf = [len(self.ebin_defs)-1]
        ks_count = []
        for k,d in enumerate(xs):
            if d-0.1 < self.cst_p['MIND']:
                if k not in kss:
                    kss.append(k)
            elif d+5.0 > self.cst_p['DCAP']:
                if k not in ksf:
                    ksf.append(k)
            else:
                ks_count.append(k)

        xs = [3.0,4.0] + [xs[k] for k in ks_count] + [40.0]
        Ps_count = [sum([self.Ps[-k-1] for k in kss])] + [self.Ps[-k-1] for k in ks_count] + [sum([self.Ps[-k-1] for k in ksf])]
        y1 = -math.log((Ps_count[0]+MEFF)/Pref)
        ys = [max(3.0,y1+3.0),y1]+ [ys[k] for k in ks_count] + [0.0] #constant fade
        
        out = open(outf,'w')
        out.write('#d0: %8.3f\n'%d0)
        out.write('#P '+'\t%7.3f'*len(Ps)%tuple([d0-D for D in self.ebin_defs])+'\n')
        out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
        out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
        out.close()

    def write_line(self):
        (res1,res2,atm1,atm2,d0) = self.distinfo
        header = 'AtomPair %3s %3d %3s %3d '
        
        l = header%(atm1,res1,atm2,res2)
        if self.csttype == 'fharm':
            l += ' FLAT_HARMONIC %5.1f %5.2f %5.2f #Pcore %6.3f\n'%tuple([d0]+self.cst_p) #sig/tol
        elif self.csttype == 'spline':
            l += ' SPLINE TAG %s 1.0 %5.2f 0.5\n'%tuple(self.cst_p) #weight
        #SOG DROPPED in this version
        
        return l
        
class Error2cst:
    def __init__(self,params=None):#,npz,pdb,cstprefix=None):
        if params == None:
            parse_opts(sys.argv[1:])
        else:
            self.opt = params
            
        self.dat1D = np.load(self.opt.npz)['lddt']
        self.dat2D = np.load(self.opt.npz)['estogram']
        self.nres = len(self.dat1D)
        self.CAcrds = myutils.pdb2crd(self.opt.pdb,'CA') #dictionary
        self.CBcrds = myutils.pdb2crd(self.opt.pdb,'CB') #dictionary
        self.aas    = myutils.pdb2res(self.opt.pdb)      #dictionary
        self.lddtG = None

        self.cstentries = [] #storage for PairCstClass instances
        
        ## Default hard-cst params
        self.hard_p = {}
        self.hard_p['PCORE'] = (0.6,0.7) # at what Pcore to apply flat-bottom harmonic?
        self.hard_p['TOL']   = (1.0,1.0) #Tolerance of flat-harmonic function
        self.hard_p['SIGMA'] = (1.0,0.2) #Sigma of flat-harmonic function
        self.hard_p['MAXD0_FHARM'] = 20.0  # Max input model's (i,j) distance to generate flat-harmonic

        ## Default soft-cst params
        self.soft_p = {}
        self.soft_p['W_SPLINE'] = 1.0 # Relative weight 
        self.soft_p['MAXD0_SPLINE'] = 35.0 # Max input model's (i,j) distance to generate spline restraints 
        self.soft_p['MIND'] = 4.0  # Minimum distance in spline x
        self.soft_p['DCAP'] = 40.0 # Maximum distance in spline x
        #self.soft_p['SOGsoften'] = 3.0 #SOG, unused
        #self.soft_p['fuzzyPcut'] = 0.25 #SOG, unused

        # Default crd-cst params
        self.crd_p = {}
        self.crd_p['LDDT_CUT'] = (0.6,0.8)
        self.crd_p['SIGMA']    = (2.0,0.2)
        self.crd_p['TOL']      = (2.0,0.5)
        
    def estimate_lddtG_from_lr(self,min_seqsep=13):
        P0mean = np.zeros(self.nres)
        for i in range(self.nres):
            n = 0
            for j in range(self.nres):
                if abs(i-j) < min_seqsep: continue ## up to 3 H turns
                n += 1
                P0mean[i] += calc_Pcore(self.dat2D[i][j]) #+-1
            P0mean[i] /= n
        return np.mean(P0mean)

    # main func for cst
    def make_pair_list_from_error(self,ulrs,ebin_defs=[]):
        if self.opt.softcst == "spline" and not os.path.exists(self.opt.splinedir):
            os.mkdir(self.opt.splinedir)

        ulrres = []
        for ulr in ulrs: ulrres += ulr
        
        #ebin_defs: how input distogram defined
        # e.g.: 
        d0mtrx = myutils.read_d0mtrx(self.opt.pdb)

        for i in range(self.nres-4):
            res1 = i+1
            atm1 = 'CB'
            if self.aas[res1]== 'GLY': atm1 = 'CA'
        
            for j in range(self.nres):
                if j-i < 4: continue
                res2 = j+1
                atm2 = 'CB'
                if self.aas[res2] == 'GLY': atm2 = 'CA'

                is_in_ulr = (res1 in  ulrres) or (res2 in ulrres)
            
                d0 = d0mtrx[i][j]
            
                seqsep = min(max(4,abs(i-j)),50)
            
                P1 = self.dat2D[i][j]
                P2 = self.dat2D[j][i]
                Pcorrect = [0.5*(P1[k]+P2[k]) for k in range(len(P1))]
                # no reference correction in this version
                Pcorrect = [P/sum(Pcorrect) for P in Pcorrect] #renormalize
            
                if d0 > self.soft_p['MAXD0_SPLINE']: continue # 35.0 ang
                if seqsep < 4: continue

                Pcore = calc_Pcore(Pcorrect)

                # hard cst
                if Pcore > self.hard_p['PCORE'][0]:
                    if Pcore > self.hard_p['PCORE'][1]: kP = 1
                    else: kP = 0
                    cst_p = [self.hard_p['SIGMA'][kP],self.hard_p['TOL'][kP],Pcore] #sig/tol/Pcore
                    cst = PairCstClass(Pcorrect,(res1,res2,atm1,atm2,d0),'fharm',cst_p) 

                    self.cstentries.append(cst)
                
                # soft cst
                if self.opt.softcst == 'spline':
                    splf="%s/%s.%d.%d.txt"%(self.opt.splinedir,self.opt.pdb,res1,res2)
                    cst_p = [splf,self.soft_p['W_SPLINE']], #filename,weight
                    cst = PairCstClass(Pcorrect,(res1,res2,atm1,atm2,d0),'spline',cst_p,
                                       ebin_defs=self.ebin_defs)
                    if not os.path.exists(splf): cst.P2spline()
                    
                    self.cstentries.append(cst)
                    
                #elif self.opt.softcst == 'sog':
                    
    def ULR_from_error(self,lddtG,fmin=0.15,fmax=0.25,dynamic=False):
        # If dynamic mode, estimate fraction min/max of ULR from pred. lddtG value
        if dynamic: #make it aggressive!
            fmax = 0.3+0.2*(0.55-lddtG)/0.3 #lddtG' range b/w 0.25~0.55
            if fmax > 0.5: fmax = 0.5
            if fmax < 0.3: fmax = 0.3
            fmin = fmax-0.1
            #print( "dynamic ULR: lddtPred/fmin/fmax: %8.5f %6.4f %6.4f"%(lddtG, fmin, fmax))

        # non-local distance accuracy -- removes bias from helices
        P0mean = np.zeros(self.nres)
        for i in range(self.nres):
            n = 0
            for j in range(self.nres):
                if abs(i-j) < 13: continue ## up to 3 H turns
                n += 1
                P0mean[i] += sum(self.dat2D[i][j][6:9]) #+-1
            P0mean[i] /= n
            
        #soften by 9-window sliding
        P0mean_soft = np.zeros(self.nres)
        for i in range(self.nres):
            n = 0
            for k in range(-4,5):
                if i+k < 0 or i+k >= self.nres: continue
                n += 1
                P0mean_soft[i] += P0mean[i+k]
            P0mean_soft[i] /= n
        P0mean = P0mean_soft

        # ULR prediction with dynamic threshold
        lddtCUT = 0.3 #initial
        for it in range(50):
            factor = 1.1
            if it > 10: factor = 1.05
        
            is_ULR = [False for ires in range(self.nres)]
            for i,lddtR in enumerate(P0mean): #lddtR: residue lddt predicted
                if lddtR < lddtCUT: is_ULR[i] = True
            myutils.sandwich(is_ULR,super_val=True,infer_val=False)
            myutils.sandwich(is_ULR,super_val=False,infer_val=True)
            f = is_ULR.count(True)*1.0/len(is_ULR)

            if f < fmin: lddtCUT *= factor
            elif f > fmax: lddtCUT /= factor
            else: break

        ULR = []
        for i,val in enumerate(is_ULR):
            if val: ULR.append(i+1)
        return myutils.trim_lessthan_3(ULR,self.nres) #single list

    def write_crdcst(self,cencst,facst):
        form = 'CoordinateConstraint CA %3d CA %3d %8.3f %8.3f %8.3f FLAT_HARMONIC 0.0  %5.3f  %5.3f #%5.3f\n'

        crd_cont = []
        for i,val in enumerate(self.dat1D):
            crd = self.CAcrds[i+1]
            if val > self.crd_p['LDDT_CUT'][1]: k = 1
            elif val > self.crd_p['LDDT_CUT'][0]: k = 0
            else: continue

            sig = self.crd_p['SIGMA'][k]
            tol = self.crd_p['TOL'][k]
            crd_cont.append(form%(i+1,i+1,crd[0],crd[1],crd[2],sig,tol,val))
            
        if len(crd_cont) >= 10: #meaningful only if there is a "core"
            cencst.writelines(crd_cont)
            facst.writelines(crd_cont)
   
    def run(self,ebin_defs=[]):
        if len(ebin_defs) == 0: #call default 
            ebin_defs = np.array([-99.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.5,0,0.5,1.5,3.0,7.0,12.5,17.5,99.0])
        if len(ebin_defs) != len(self.dat2D[0][0]):
            sys.exit('Error bin definition differs from input 2D error data!')

        lddtG_lr = self.estimate_lddtG_from_lr() #lddt, long-range only
        lddtG    = np.mean(self.dat1D)           #lddt, regular
        
        self.ulrs_conserve = self.ULR_from_error(lddtG_lr,fmin=0.10,fmax=0.20) #conserv. pred.
        self.ulrs_dynamic  = self.ULR_from_error(lddtG_lr,dynamic=True)        #dynamic pred.
        if self.opt.ulr_pred:
            ulrs = self.ulrs_dynamic
        else:
            ulrs = self.opt.ulrs_static
            
        self.make_pair_list_from_error(ulrs,ebin_defs) #make full list of pair csts

        # write 
        cencst = open(self.opt.prefix+'.cst','w')
        facst = open(self.opt.prefix+'.fa.cst','w')

        # first write paircst to both cen & fa
        for cst in self.cstentries:
            l = cst.write_line()
            cencst.write(l)
            if cst.csttype == 'spline': continue #not to facst
            facst.write(l)
        # next write coord cst
        if self.opt.crdcst and lddtG >= self.crd_p['LDDT_CUT'][0]:
            print( "Adding coordinate cst...")
            self.write_crdcst(cencst,facst)

        cencst.close()
        facst.close()
        
if __name__ == "__main__":
    a = Error2cst()
    a.run()
