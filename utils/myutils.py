import sys,os,copy
from math import sqrt
import numpy as np

###### simple math to replace someday....
def normalize(v1):
    val=0.0
    for val in v1: val+=val*val
    for i in range(len(v1)):
        v1[i]/=val**0.5
    return v1

def distance(crd1,crd2):
    dcrd = [crd1[k]-crd2[k] for k in range(3)]
    return sqrt(dcrd[0]*dcrd[0] + dcrd[1]*dcrd[1] + dcrd[2]*dcrd[2])

def d2(mol1crd,mol2crd):
    displ=[]
    for i in range(3):
        displ.append(abs(float(mol1crd[i])-float(mol2crd[i])))
    return displ[0]**2+displ[1]**2+displ[2]**2

def inproduct(v1,v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

def cross(v1,v2):
    rturn [v1[1]*v2[2]-v1[2]*v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]]

####### PDB parser
        
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

def aa1toaa3(aa1):
    return {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
            'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
            'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}[aa1]

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
        if isinstance(comp,int) or isinstance(comp,np.int64):
            if i == 0 or abs(comp-prv) != 1:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        elif isinstance(comp,str):
            if i == 0 or comp != prv:
                partlist.append([comp])
            else:
                partlist[-1].append(comp)
        else:
            print("comp is neither int nor str: ",type(comp))
            sys.exit()
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

########## external stuffs

def SS_fromdssp(pdbfile,log=''):
    if log == '':
        cont = os.popen('/home/hpark/util/dsspcmbi -na %s 2> /dev/null'%pdbfile).readlines()
    else:
        cont = open(log).readlines()
    SSdict={}
    SS3dict={}
    read_cont = False
    for i,line in enumerate(cont):
        if 'RESIDUE AA' in line:
            read_cont = True
            continue
        if not read_cont:
            continue

        if line[5:10] == '     ':
            continue
        (resno,SS)=(int(line[5:10]),line[16])
        SSdict[resno] = SS
        SS3dict[resno] = dssp2psipred(SS)
    return SSdict, SS3dict

def SS_frompross(pdbfile):
    cont = os.popen('~/util/pross.py %s'%pdbfile).readlines()[2:]
    SS = {}
    for line in cont:
        linesp = line[:-1].split()
        try:
            resno = int(linesp[0])
        except:
            continue
        SS[resno] = linesp[2]
    return SS

def get_naccess_rsa(pdbfile,mode='rsa',atom='sc',overwrite=False):
    rsafile = '%s.rsa'%(pdbfile.split('/')[-1].split('.')[0])
    if not os.path.exists(rsafile) or overwrite:
        #os.system('/work/hpark/bin/naccess %s 1> /dev/null'%pdbfile)
        os.system('/home/hpark/bin/freesasa %s  --format=rsa > %s'%(pdbfile,rsafile))
        print('/home/hpark/bin/freesasa %s  --format=rsa > %s'%(pdbfile,rsafile))
        #print ('~/bin/naccess %s 1> /dev/null'%pdbfile)
    sa = read_rsa(rsafile,mode=mode,atom=atom)
    return sa

def betainfo_fromdssp(pdbfile,min_strand_len=0):
    cont = os.popen('/home/hpark/util/dsspcmbi -na %s 2> /dev/null'%pdbfile).readlines()
    
    read_cont = False
    extres = []
    coils = []
    respairs = []
    npaired = {}
    paired_res = {}
    resmap = {}
    for i,line in enumerate(cont):
        if 'RESIDUE AA' in line:
            read_cont = True
            continue
        if not read_cont:
            continue

        if line[5:10] == '     ':
            continue
        (resno,SS)=(int(line[5:10]),line[16])
        if SS in ['E']:
            extres.append(resno)
        elif SS in ['C',' ','S','B','L']: # don't include turn
            coils.append(resno)

        resmap[int(line[:5])] = resno

        pairres1 = int(line[25:29])
        pairres2 = int(line[29:33])
        npaired[resno] = 0
        paired_res[resno] = []
        if pairres1 != 0:
            respairs.append([resno,pairres1])
            npaired[resno] += 1
            paired_res[resno].append(pairres1)
        if pairres2 != 0:
            respairs.append([resno,pairres2])
            npaired[resno] += 1
            paired_res[resno].append(pairres2)

    # remap paired res
    for resno in paired_res:
        for i,pairres in enumerate(paired_res[resno]):
            paired_res[resno][i] = resmap[pairres]

    strands = list2part(extres)
    for str1 in copy.copy(strands):
        if len(str1) < min_strand_len:
            strands.remove(str1)
            coils += str1

    strand_npaired = []
    for str1 in strands:
        strand_npaired.append([])
        for res in str1:
            strand_npaired[-1].append(npaired[res])

    pairings = []
    if len(strands) > 1:
        for i,str1 in enumerate(strands[:-1]):
            for str2 in strands[i+1:]:
                j = strands.index(str2)
                if is_pairing(str1,str2,respairs):
                    pairings.append([i,j])
    #return strands, strand_npaired, pairings
    return strands, coils, paired_res, pairings

def write_as_pdb(outf,crds,chainno=[]):
    out= open(outf,'w')
    l = "ATOM     %2d  %-2s  UNK %s  %2d    %8.3f%8.3f%8.3f  1.00  0.00           %s\n"
    j = 0
    for i,crd in enumerate(crds):
        chain = 'A'
        if len(chainno) == len(crds):
            chain = 'ABCDEFGHIJKLMNOPQ'[chainno[i]]
        
        if len(crd) == 3:
            out.write(l%(j,"CA",chain,i,crds[i][0],crds[i][1],crds[i][2],'C'))
            j += 1
        else:
            out.write(l%(j  ,"N", chain,i,crds[i][0][0],crds[i][0][1],crds[i][0][2],'N'))
            out.write(l%(j+1,"CA",chain,i,crds[i][1][0],crds[i][1][1],crds[i][1][2],'C'))
            out.write(l%(j+2,"C" ,chain,i,crds[i][2][0],crds[i][2][1],crds[i][2][2],'C'))
            out.write(l%(j+3,"O" ,chain,i,crds[i][3][0],crds[i][3][1],crds[i][3][2],'O'))
            #out.write(l%(j+4,"CB" ,i,crds[i][4][0],crds[i][4][1],crds[i][4][2],'C'))
            j+=4
    out.close()

def blosum62(res1,res2):
    aaid = ['A','R','N','D','Q','C','E','G','H','I',
            'L','K', 'M','F','P','S','T','W','Y','V','B','J','Z']

    mtrx = [[] for k in range(23)]
    mtrx[0]  = (  4,-1,-2,-2,-1, 0,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1,-1)
    mtrx[1]  = ( -1, 5, 0,-2, 1,-3, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-2, 0)
    mtrx[2]  = ( -2, 0, 6, 1, 0,-3, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 4,-3, 0)
    mtrx[3]  = ( -2,-2, 1, 6, 0,-3, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4,-3, 1)
    mtrx[4]  = ( -1, 1, 0, 0, 5,-3, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0,-2, 4)
    mtrx[5]  = (  0,-3,-3,-3,-3, 9,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-1,-3)
    mtrx[6]  = ( -1, 0, 0, 2, 2,-4, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1,-3, 4)
    mtrx[7]  = (  0,-2, 0,-1,-2,-3,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-4,-2)
    mtrx[8]  = ( -2, 0, 1,-1, 0,-3, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0,-3, 0)
    mtrx[9]  = ( -1,-3,-3,-3,-3,-1,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3, 3,-3)
    mtrx[10] = ( -1,-2,-3,-4,-2,-1,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4, 3,-3)
    mtrx[11] = ( -1, 2, 0,-1, 1,-3, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0,-3, 1)
    mtrx[12] = ( -1,-1,-2,-3, 0,-1,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3, 2,-1)
    mtrx[13] = ( -2,-3,-3,-3,-3,-2,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3, 0,-3)
    mtrx[14] = ( -1,-2,-2,-1,-1,-3,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-3,-1)
    mtrx[15] = (  1,-1, 1, 0, 0,-1, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0,-2, 0)
    mtrx[16] = (  0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1,-1)
    mtrx[17] = ( -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-2,-2)
    mtrx[18] = ( -2,-2,-2,-3,-1,-2,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-1,-2)
    mtrx[19] = (  0,-3,-3,-3,-2,-1,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3, 2,-2)
    mtrx[20] = ( -2,-1, 4, 4, 0,-3, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4,-3, 0)
    mtrx[21] = ( -1,-2,-3,-3,-2,-1,-3,-4,-3, 3, 3,-3, 2, 0,-3,-2,-1,-2,-1, 2,-3, 3,-3)
    mtrx[22] = ( -1, 0, 0, 1, 4,-3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-2,-2,-2, 0,-3, 4)

    if len(res1) == 3:
        aa1 = threecode_to_alphabet(res1)
        aa2 = threecode_to_alphabet(res2)
    else:
        aa1 = res1
        aa2 = res2
    if aa1 not in aaid or aa2 not in aaid:
        return False
    i1 = aaid.index(aa1)
    i2 = aaid.index(aa2)
    return mtrx[i1][i2]

