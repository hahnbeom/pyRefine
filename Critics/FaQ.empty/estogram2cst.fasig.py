import sys,os,copy
import numpy as np
from math import log,sqrt

CURR = os.getcwd()

MEFF=0.0001
## THIS IS ERROR ESTIMATION! SUGGESTION SHOULD BE INVERSE
#DELTA = [-99.0,-17.5,-12.5,-7.5,-3.75,-1.25,0, 
#         1.25,3.75,7.5,12.5,17.5,99.0]
DELTA = [-99.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.75,
         0.0,
         0.75,1.5,3.0,7.0,12.5,17.5,99.0]
#REFSTAT_FILE = '/home/hpark/projects/ML/decoys9000/make_refstat_models/pred.v4sm/refstat.dbin.seqsep.txt'
MYPATH = os.path.dirname(os.path.realpath(__file__))
REFSTAT_FILE = MYPATH + '/refstat.dbin.seqsep.Q.txt'

##############################################
MAXD0_SPLINE = 35.0 # Max input model's (i,j) distance to generate spline restraints 
MIND = 4.0  # Minimum distance in spline x
DCAP = 40.0 # Maximum distance in spline x

FUNC = ['bounded','sigmoid'] #cen/fa for Pcen > PCORE[1]

PCON_BOUNDED = 0.9
MAXD0_FHARM = 20.0  # Max input model's (i,j) distance to generate flat-harmonic
TOL = [3.0,4.0,2.0,1.0] #Tolerance of flat-harmonic function
PCORE=[0.3,0.6,0.7,0.8] # at what P(abs(deltaD)<1) to apply flat-bottom harmonic? (will skip flat-bottom harmonic if lower then this value)
P_spline_on = 1.0 # never use
Sigma=[2.5,2.5,2.0,1.0] #Sigma of flat-harmonic function
kCEN = int((len(DELTA)-1)/2)

###############################################
# Misc.
form_spl = 'SPLINE TAG %-30s 1.0 1.0 0.5 #%s\n' #W_SPLINE = 1.0
def param2cst(tol,sig,d0,Pstr,func,extra='',w_relative=False):
    form_h = 'FLAT_HARMONIC %5.1f %5.2f %5.2f #%s\n'
    form_b = 'BOUNDED %5.1f %5.1f 1.0 %5.2f #%s\n'
    form_sig = 'SUMFUNC 3 SCALARWEIGHTEDFUNC %6.3f SIGMOID %6.3f %6.3f SCALARWEIGHTEDFUNC %6.3f  SIGMOID %6.3f %6.3f CONSTANTFUNC %6.3f #%s\n' #(w,x1,m1,-w,x2,m2,2*w) == (5/t,t,s,-5/t,t,s,10/t)

    if func == 'bounded':
        funcstr = form_b%(d0-tol,d0+tol,sig,Pstr)
    elif func == 'fharm':
        funcstr = form_h%(d0-tol,d0+tol,sig,Pstr)
    elif func == 'sigmoid':
        w = 1.0
        if w_relative: w = 2.0/sig
        m = 5.0/sig
        x1 = d0-tol
        x2 = d0+tol
        funcstr = form_sig%(-w,x1,m,w,x2,m,w,Pstr) #(w,x1,m1,-w,x2,m2,shift)
    elif func == 'spline':
        funcstr = form_spl%(extra,Pstr)
    
    return funcstr

def pdb_in_resrange(pdb,newname,resrange,skip_alt=True,exres=False):
    cont = open(pdb)
    newpdb = open(newname,'w')
    newcont = []
    for line in cont:
        if line[:4] not in ['ATOM','HETA']:
            continue
        resno = int(line[22:26])
        alt = line[16]
        if skip_alt and (alt not in [' ','A']):
            continue
        if exres:
            if resno not in resrange:
                newcont.append(line)
        else:
            if resno in resrange:
                newcont.append(line)
    newpdb.writelines(newcont)

def pdb2crd(pdbfile,opt,res_in=[],ignore_insertion=False,chaindef=[]):
    pdbcont=open(pdbfile)
    res_defined = False
    crd={}

    for line in pdbcont:
        if line[:4]!='ATOM':
            continue
        resno = int(line[22:26])
        restype = line[16:20].strip()
        chain = line[21]
        if ignore_insertion and line[26] != ' ':
            continue
        if res_defined and resno not in res_in:
            continue
        if chaindef != [] and chain not in chaindef:
            continue

        atmtype = line[12:16].strip()
        if opt == 'CA':
            if line[12:16] == ' CA ':
                if resno in crd:
                    continue
                crd[resno] = [float(line[30+i*8:38+i*8]) for i in range(3)]
        elif opt == 'CB':
            if (restype == 'GLY' and line[12:16] == ' CA ') \
                   or line[12:16] == ' CB ':
                if resno in crd:
                    continue
                crd[resno] = [float(line[30+i*8:38+i*8]) for i in range(3)]
        else:
            if resno not in crd:
                crd[resno] = {}
            crd[resno][atmtype] = [float(line[30+i*8:38+i*8]) for i in range(3)]
    pdbcont.close()
    return crd

def pdb2res(pdbfile,bychain=False,chaindef=[],withchain=False,single=False):
    pdbcont = open(pdbfile)
    restype = {}
    for line in pdbcont:
        if line[:4]!='ATOM':
            continue

        if line[12:16].strip() == 'CA':
            res = int(line[22:26])
            chain = line[21]
            if withchain:
                res = '%s%04d'%(chain,res)

            if line[26] != ' ':
                continue
            if chaindef != [] and chain not in chaindef:
                continue

            char = line[17:20]
            if char in ['HIP','HID','HIE']:
                char = 'HIS'
            elif char == 'CSS':
                char = 'CYS'
            if single:
                char = threecode_to_alphabet(char)

            if bychain:
                if chain not in restype:
                    restype[chain] = {}
                restype[chain][res] = char
            else:
                restype[res] = char
    return restype

def distance(crd1,crd2):
    dcrd = [crd1[k]-crd2[k] for k in range(3)]
    return sqrt(dcrd[0]*dcrd[0] + dcrd[1]*dcrd[1] + dcrd[2]*dcrd[2])

def aa3toaa1(aa3):
    return {'ALA':'A','CYS':'C','CYD':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','MSE':'M','CSO':'C','SEP':'S'}[aa3]

def read_d0mtrx(pdb,nres=-1,byresno=False):
    crds = pdb2crd(pdb,'CB')
    reslist = list(crds.keys())
    reslist.sort()
    if nres == -1: nres = len(reslist)

    d0mtrx = [[0.0 for i in range(nres)] for j in range(nres)]
    for i,ires in enumerate(reslist):
        for j,jres in enumerate(reslist):
            if byresno:
                d0mtrx[ires-1][jres-1] = distance(crds[ires],crds[jres])
                d0mtrx[jres-1][ires-1] = d0mtrx[ires-1][jres-1]
            else:
                d0mtrx[i][j] = distance(crds[ires],crds[jres])
                d0mtrx[j][i] = d0mtrx[i][j]
    return d0mtrx

######################################
# ULR
def list2part(inlist):
    partlist = []
    for i,comp in enumerate(inlist):
        if isinstance(comp,int):
            if i == 0 or abs(comp-prv) != 1:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        elif isinstance(comp,str):
            if i == 0 or comp != prv:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        prv = comp
    return partlist

def part2list(inlist):
    conlist = []
    for reg in inlist:
        conlist += reg
    return conlist

def trim_lessthan_3(ulrin,nres):
    regs = list2part(ulrin)
    for reg in copy.copy(regs):
        if len(reg) < 3:
            regs.remove(reg)
            
    ulrres = []
    for i,reg in enumerate(regs[:-1]):
        if reg[0] <= 3:
            reg = list(range(1,reg[-1]+1))
        if regs[i+1][0]-reg[-1] <= 3:
            reg += list(range(reg[-1]+1,regs[i+1][0]))
        ulrres += reg

    if nres-regs[-1][-1] <= 3:
        regs[-1] = range(regs[-1][0],nres+1)
    ulrres += regs[-1]
    return ulrres

def ulr2trim(ulr,nres):
    totrim = []
    ulr_by_reg = list2part(ulr)
    
    for reg in ulr_by_reg:
        res1 = min(reg)
        res2 = max(reg)
        #if abs(res2-res1) > 5:
        #    if res2 < nres: res2 -= 2
        #    if res1 > 1: res1 += 2
        if abs(res2-res1) > 3:
            if res2 < nres: res2 -= 1
            if res1 > 1: res1 += 1
        totrim += range(res1,res2+1)
    return totrim

def sandwich(inlist,super_val,infer_val):
    inlist_cp = copy.copy(inlist)
    for i,val in enumerate(inlist_cp[:-1]):
        if i == 0:
            continue
        if inlist_cp[i-1] == super_val and inlist_cp[i+1] == super_val:
            inlist[i] = super_val

    inlist_cp = copy.copy(inlist)
    for i,val in enumerate(inlist_cp[:-1]):
        if i == 0:
            continue
        if inlist_cp[i-1] == infer_val and inlist_cp[i+1] == infer_val:
            inlist[i] = infer_val

### not working properly...
def pick_ulr_recursive(pred,Pcut,is_ULR):
    P0mean = pred2P0mean(pred)
    P0all = sum(P0mean)/(is_ULR.count(False)+0.1)
    #print is_ULR.count(True), P0all, Pcut
    it = 0
    while P0all < Pcut:
        for i,P in enumerate(P0mean):
            if P < Pcut: is_ULR[i] = True
        P0mean = pred2P0mean(pred,is_ULR)
        P0all = sum(P0mean)/(0.1+is_ULR.count(False))
        it += 1
        #print it, is_ULR.count(True), P0all, Pcut

def estimate_lddtG_from_lr(pred):
    nres = len(pred)
    P0mean = np.zeros(nres)
    for i in range(nres):
        n = 0
        for j in range(nres):
            if abs(i-j) < 13: continue ## up to 3 H turns
            n += 1
            P0mean[i] += sum(pred[i][j][6:9]) #+-1
        P0mean[i] /= n
    return np.mean(P0mean)
        
# MAIN modification as using lddt pred as input now
def ULR_from_pred(pred,lddtG,fmin=0.15,fmax=0.25,dynamic=False,mode='mean'):
    nres = len(pred) #pred is lddt-per-res
    if dynamic: #make it aggressive!
        fmax = 0.3+0.2*(0.55-lddtG)/0.3 #lddtG' range b/w 0.25~0.55
        if fmax > 0.5: fmax = 0.5
        if fmax < 0.3: fmax = 0.3
        fmin = fmax-0.1
        print( "dynamic ULR: lddtPred/fmin/fmax: %8.5f %6.4f %6.4f"%(lddtG, fmin, fmax))

    # non-local distance accuracy -- removes bias from helices
    P0mean = np.zeros(nres)
    for i in range(nres):
        n = 0
        for j in range(nres):
            if abs(i-j) < 13: continue ## up to 3 H turns
            n += 1
            P0mean[i] += sum(pred[i][j][6:9]) #+-1
        P0mean[i] /= n
    #soften by 9-window sliding
    P0mean_soft = np.zeros(nres)
    for i in range(nres):
        n = 0
        for k in range(-4,5):
            if i+k < 0 or i+k >= nres: continue
            n += 1
            P0mean_soft[i] += P0mean[i+k]
        P0mean_soft[i] /= n
        #print( "%3d %8.4f %8.4f"%(i, P0mean[i], P0mean_soft[i]) )
        
    P0mean = P0mean_soft
        
    lddtCUT = 0.3 #initial
    for it in range(50):
        factor = 1.1
        if it > 10: factor = 1.05
        
        is_ULR = [False for ires in range(nres)]
        if mode == 'mean':
            for i,lddtR in enumerate(P0mean):
                if lddtR < lddtCUT: is_ULR[i] = True
        #elif mode == 'recursive':
        #    pick_ulr_recursive(pred,lddtCUT,is_ULR)
            
        sandwich(is_ULR,super_val=True,infer_val=False)
        sandwich(is_ULR,super_val=False,infer_val=True)
        f = is_ULR.count(True)*1.0/len(is_ULR)

        if f < fmin: lddtCUT *= factor
        elif f > fmax: lddtCUT /= factor
        else: break

    ULR = []
    for i,val in enumerate(is_ULR):
        if val: ULR.append(i+1)
    return trim_lessthan_3(ULR,nres)
   
###################################################
# Spline

def read_Pref(txt):
    Pref = {}
    refD = 0
    for l in open(txt).readlines():
        words = l[:-1].split()
        if len(words) == len(DELTA)+3: #dbin only
            ibin = int(words[0])
            #d = int(words[1])
            Ps = [float(word) for word in words[3:]]
            Pref[ibin] = Ps
            refD = 1
        elif len(words) == len(DELTA)+4: #seqsep
            ibin = int(words[0])
            if ibin not in Pref: Pref[ibin] = {}
            seqsep = int(words[2])
            Ps = [float(word) for word in words[4:]]
            Pref[ibin][seqsep] = Ps
            refD = 2
        elif len(words) == len(DELTA)+5: #seqsep & Q
            ibin = int(words[0])
            seqsep = int(words[2])
            Qbin = int(words[3])
            if ibin not in Pref: Pref[ibin] = {}
            if seqsep not in Pref[ibin]: Pref[ibin][seqsep] = {}
            Ps = [float(word) for word in words[5:]]
            Pref[ibin][seqsep][Qbin] = Ps
            refD = 3
    return Pref,refD
    
def P2spline(outf,d0,Ps,Ps_uncorrected):
    xs = []
    ys = []
    Pref = Ps[kCEN] 
    Ps_count = []
    Ps_count_u = []

    for k,P in enumerate(Ps):#[1:-1]):
        d = d0 - DELTA[k]
        E = -log((P+MEFF)/Pref)
        xs.append(d)
        ys.append(E)

    xs.reverse()
    ys.reverse()

    kss = [0]
    ksf = [len(DELTA)-1]
    ks_count = []
    for k,d in enumerate(xs):
        if d-0.1 < MIND:
            if k not in kss:
                kss.append(k)
        elif d+5.0 > DCAP:
            if k not in ksf:
                ksf.append(k)
        else:
            ks_count.append(k)

    xs = [3.0,4.0] + [xs[k] for k in ks_count] + [40.0]
    Ps_count = [sum([Ps[-k-1] for k in kss])] + [Ps[-k-1] for k in ks_count] + [sum([Ps[-k-1] for k in ksf])]
    Ps_count_u = [sum([Ps_uncorrected[-k-1] for k in kss])] + [Ps_uncorrected[-k-1] for k in ks_count] + [sum([Ps_uncorrected[-k-1] for k in ksf])]
    y1 = -log((Ps_count[0]+MEFF)/Pref)
    ys = [max(3.0,y1+3.0),y1]+ [ys[k] for k in ks_count] + [0.0] #constant fade
        
    out = open(outf,'w')
    out.write('#d0: %8.3f\n'%d0)
    out.write('#P '+'\t%7.3f'*len(Ps)%tuple([d0-D for D in DELTA])+'\n')
    out.write('#P '+'\t%7.5f'*len(Ps)%tuple(Ps)+'\n')
    out.write('#Pu'+'\t%7.5f'*len(Ps_uncorrected)%tuple(Ps_uncorrected)+'\n')
    out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
    out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
    out.close()

# main func for cst
def estogram2cst(dat,pdb,cencst,facst,
                 weakcst,
                 ulr=[],
                 do_reference_correction=False, # use it only for sm version,
                 Pcore=PCORE,func=FUNC,
                 Anneal=1.0,w_relative=False
):
    if (weakcst=="spline") and not os.path.exists('splines'):
        os.mkdir('splines')

    Q = np.mean(dat['lddt'])
    dat = dat['estogram']
    nres = len(dat)
    d0mtrx = read_d0mtrx(pdb)
    aas = pdb2res(pdb)
    Prefs = None
    
    # DO NOT USE REFERENCE CORRECTION BELOW UNLESS YOUR NPY DATA IS FROM THE LATEST SINGLE-MODEL VERSION
    if do_reference_correction:
        Prefs,refD = read_Pref(REFSTAT_FILE) #v3sm
        #Qbins = [0.4,0.5,0.6,0.7,0.8,1.0]
        Qbin = min(5,max(0,int(Q-0.4)/0.1))

    nharm_cst_lr = 0
    soft_cst_info = []
    pdbprefix = pdb.split('/')[-1][:-4]
    
    for i in range(nres-4):
        res1 = i+1
        atm1 = 'CB'
        if aas[res1]== 'GLY': atm1 = 'CA'
        
        for j in range(nres):
            if j-i < 4: continue
            res2 = j+1
            #is_in_ulr = (res1 in ulr) or (res2 in ulr)
            is_in_ulr = False
            for k in range(-2,3):
                if res1+k in ulr:
                    is_in_ulr = True
                    break
                if res2+k in ulr:
                    is_in_ulr = True
                    break
            
            d0 = d0mtrx[i][j]
            
            #seqsep = min(max(4,abs(i-j)),50)
            seqsep = min(50,abs(i-j))
            if seqsep > 20:
                seqsep = int(20+(seqsep-20)/2) #trimmed version

            P1 = dat[i][j]
            P2 = dat[j][i]
            Pavrg = [0.5*(P1[k]+P2[k]) for k in range(len(P1))]

            if do_reference_correction:
                dbin = int(d0-4.0)
                if dbin <0: dbin = 0
                elif dbin >= max(Prefs.keys()): dbin = max(Prefs.keys())

                if refD == 1: #seqsep not in Prefs[dbin]:
                    if seqsep < 10:
                        Pref = Prefs[dbin][kCEN]
                    else:
                        sys.exit("No Pref at dbin/seqsep=%d/%d"%(dbin,seqsep))
                elif refD == 2:
                    Pref = Prefs[dbin][seqsep]
                elif refD == 3:
                    Pref = Prefs[dbin][seqsep][Qbin]
                    
                Pcorrect = [P/(Pref[k]+0.001) for k,P in enumerate(Pavrg)]
            else:
                Pcorrect = Pavrg
                
            Pcorrect = [P/sum(Pcorrect) for P in Pcorrect] #renormalize
            
            if d0 > MAXD0_SPLINE: continue # 35.0 ang
            
            atm2 = 'CB'
            if aas[res2] == 'GLY': atm2 = 'CA'
            if seqsep < 4: continue

            #dexp = d0
            #is_extra_contact = False
            #for k,P in enumerate(Pcorrect):
            #    dexp -= DELTA[k]*P
            #if dexp < 10.0:
            #    is_extra_contact = True

            Pcontact = 0.0
            for k,P in enumerate(Pcorrect):
                if d0 - DELTA[k] < 8.0:
                    Pcontact += P
                elif d0 - DELTA[k] < 10.0:
                    Pcontact += 0.5*P

            # sum of 3-contiguous P after correction
            maxP = max([sum(Pcorrect[k:k+2]) for k in range(1,len(DELTA)-1)])

            Pcen = np.sum(Pavrg[kCEN-1:kCEN+2]) #from uncorrected

            aa1 = aa3toaa1(aas[res1])
            aa2 = aa3toaa1(aas[res2])
            cstheader = 'AtomPair %3s %3d %3s %3d '%(atm1,res1,atm2,res2)
            
            #Spline
            if weakcst == 'spline':
                splf="./splines/%s.%d.%d.txt"%(pdbprefix,res1,res2)
                if not os.path.exists(splf): 
                    P2spline(splf,d0,Pcorrect,Pavrg) #always generate
                #spline for every pair < 35.0 Ang
                censtr = cstheader + param2cst(0,0,0,"Pmax %6.3f"%maxP,'spline',extra=splf)
                cencst.write(censtr)

                if maxP > P_spline_on: # confident ones only in fa
                    facst.write(cstheader+form_spl%(splf,'%5.2f'%maxP) )                 
            elif weakcst == 'bounded' and Pcontact > PCON_BOUNDED and Pcen < Pcore[1]:
                censtr = cstheader + param2cst(4.0,1.0,8.0,"Pcon %6.3f"%Pcontact,'bounded')
                cencst.write(censtr)
                
            #core part: flat bottom harmonic
            if d0 > MAXD0_FHARM or is_in_ulr:
                continue

            if Pcen > Pcore[1]:
                if Pcen > Pcore[3]:
                    censtr = cstheader + param2cst(TOL[3],Sigma[3],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[3],Sigma[3],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                elif Pcen > Pcore[2]:
                    censtr = cstheader + param2cst(TOL[2],Sigma[2],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[2],Sigma[2],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                elif Pcen > Pcore[1]:
                    censtr = cstheader + param2cst(TOL[1],Sigma[1],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[1],Sigma[1],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                    
                cencst.write(censtr)
                facst.write(fastr)
                if seqsep >= 9:
                    nharm_cst_lr += 1
                    
            elif Pcen > Pcore[0] and seqsep >= 9:
                # store as list and not apply yet
                censtr = cstheader + param2cst(TOL[0],Sigma[0],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                fastr  = cstheader + param2cst(TOL[0],Sigma[0],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                soft_cst_info.append((Pcen,censtr,fastr))

    npair_cut = log(float(nres))*float(nres)
    if nharm_cst_lr < npair_cut:
        soft_cst_info.sort()
        nadd = 0
        #try npair_cut*2: very likely to destroy topology unless highly restrained
        while nharm_cst_lr < npair_cut*2 and len(soft_cst_info) > 0: 
            (P,censtr,fastr) = soft_cst_info.pop()
            cencst.write(censtr)
            facst.write(fastr)
            nharm_cst_lr += 1
            nadd += 1
        print( 'Not enough lr cst %d (cut %d): supplement with lower-Pcen %d csts'%(nharm_cst_lr-nadd, 
                                                                                    npair_cut,
                                                                                    nharm_cst_lr))
            
# main func for ulr & partial pdb
def dat2ulr(pdb,pred,lddtG):
    outprefix = pdb.split('/')[-1].replace('.pdb','')
    
    nres = len(pred)

    ulrs = [[] for k in range(3)]
    ulrs[0] = ULR_from_pred(pred,lddtG,fmin=0.10,fmax=0.20) 
    ulrs[1] = ULR_from_pred(pred,lddtG,fmin=0.40,fmax=0.50) #unused
    ulrs[2] = ULR_from_pred(pred,lddtG,dynamic=True)

    totrim = [[] for k in range(3)]
    for k,ulr in enumerate(ulrs):
        if ulr != []:
            totrim[k] = ulr2trim(ulr,nres)
    
    print( "ULR P.std : %s %d"%(pdb,len(ulrs[0])), ulrs[0] )
    pdb_in_resrange(pdb,'partial.%s.cons.pdb'%outprefix,totrim[0],exres=True)

    print( "ULR P.aggr: %s %d"%(pdb,len(ulrs[2])), ulrs[2] ) #== dynamic mode
    pdb_in_resrange(pdb,'partial.%s.aggr.pdb'%outprefix,totrim[2],exres=True)

    return ulrs

def lddt2crdcst(pred,pdb,cut=(0.6,0.8),
                sig=(2.0,1.0),
                tol=(2.0,1.0)):

    crds = pdb2crd(pdb,'CA')
    form = 'CoordinateConstraint CA %3d CA %3d %8.3f %8.3f %8.3f FLAT_HARMONIC 0.0  %5.3f  %5.3f #%5.3f\n'
    
    crd_cont = []
    for i,val in enumerate(pred):
        crd = crds[i+1]
        if val > cut[1]:
            crd_cont.append(form%(i+1,i+1,crd[0],crd[1],crd[2],tol[1],sig[1],val))
        elif val > cut[0]:
            crd_cont.append(form%(i+1,i+1,crd[0],crd[1],crd[2],tol[0],sig[0],val))
            
    if len(crd_cont) < 10: #meaningful only if there is a "core"
        crd_cont = []
    print("Put %d coordinate restraints!"%(len(crd_cont)))
    return crd_cont
        
def main(npz,pdb,cstprefix=None):
    dat = np.load(npz)
    lddtG = np.mean(dat['lddt'])
    lddtG_lr = estimate_lddtG_from_lr(dat['estogram']) #this is long-range-only lddt
    ulrs = dat2ulr(pdb,dat['estogram'],lddtG_lr)
    
    ulr_exharm,weakcst,refcorr = ([],'none',False)
    if '-exulr_from_harm' in sys.argv:
        ulr_exharm = ulrs[2]
    if '-weakcst' in sys.argv:
        weakcst = sys.argv[sys.argv.index('-weakcst')+1]
    if '-reference_correction' in sys.argv:
        refcorr = True
        
    if cstprefix == None:
        cstprefix = pdb.replace('.pdb','')
        
    w_relative = False
    if '-w_relative' in sys.argv:
        w_relative = True

    if '-pcore' in sys.argv:
        Pcore_in = [0.4,
                    float(sys.argv[sys.argv.index('-pcore')+1]),
                    float(sys.argv[sys.argv.index('-pcore')+2]),
                    float(sys.argv[sys.argv.index('-pcore')+3])]
    else:
        Pcore_in = PCORE

    if '-func' in sys.argv:
        func_in = [sys.argv[sys.argv.index('-func')+1],
                   sys.argv[sys.argv.index('-func')+2]]
    else:
        func_in = FUNC

    if '-anneal' in sys.argv:
        Anneal_in = float(sys.argv[sys.argv.index('-anneal')+1])
    else:
        Anneal_in = 1.0

    cencst = open(cstprefix+'.cst','w')
    facst = open(cstprefix+'.fa.cst','w')
    #print( "Reporting restraints as %s & %s"%(cencst,facst))

    estogram2cst(dat,pdb,cencst,facst,
                 weakcst,#add_spline=(weakcst=='spline'),
                 ulr=ulr_exharm, #exclude fharm on dynamic-ulr
                 do_reference_correction=refcorr,
                 Pcore=Pcore_in,
                 func=func_in,
                 Anneal=Anneal_in,
                 w_relative=w_relative)

    #weakly binded
    '''
    if weakcst=='sog':
        print( "Converting weak cst into SOG func...")
        #cstcont = estogram2sog.main(npz,pdb)
        a = estogram2sog.DataClass()
        a.soften = 3.0 #control parameter
        a.fuzzyPcut = 0.3
        a.read_data(pdb,dat['estogram'])
        a.run(ncores=1) ##control

        cstcont = a.write()
        cencst.writelines(cstcont) #only for cencst
    '''
    
    if '-force_crdcst' in sys.argv:
        print( "Adding coordinate cst...")
        cstcont = lddt2crdcst(dat['lddt'],pdb,cut=(0.6,0.7))
        cencst.writelines(cstcont)
        facst.writelines(cstcont)

    cencst.close()
    facst.close()
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print( "usage: python estogram2cst.py [npy file] [reference pdb file]" )
        sys.exit()
        
    npz = sys.argv[1]
    pdb = sys.argv[2]
    outprefix = sys.argv[3]
    main(npz,pdb,outprefix)
