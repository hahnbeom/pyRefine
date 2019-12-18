import sys,copy,glob,os
from scipy.linalg import det, eig, svd, inv
from scipy import matrix, array, transpose
from math import cos,sin,sqrt
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/../../utils'%SCRIPTPATH)
from myutils import pdb2crd, pdb2res, list2part, aa1toaa3, blosum62, betainfo_fromdssp
from myutils import inproduct, d2, normalize

#RMSD_MATCH_CUT = 2.0
#FRAME_CA = True
MINRES_FOR_FRAME = 5
SEQSCORECUT=-1 # everything; use positive value to apply cut
CLASHCUT=1 # reject if any one res clashes

def rmsd_simple(crds1,crds2):
    msd = 0.0
    for i,crd1 in enumerate(crds1):
        crd2 = crds2[i]
        msd += d2(crd1,crd2)
    msd/= len(crds1)
    return sqrt(msd)

def Kabsch(crd1_in,crd2,w=[]):
    com1 = [0.0,0.0,0.0]
    com2 = [0.0,0.0,0.0]
    n = len(crd1_in)
    if w == []: w = [1.0 for i in range(n)]

    crd1 = copy.deepcopy(crd1_in)
    # calc COM and subtract it from crds
    for i in range(n):
        for k in range(3):
            com1[k] += w[i]*crd1[i][k]
            com2[k] += w[i]*crd2[i][k]
    for k in range(3):
        com1[k] /= n
        com2[k] /= n

    for i in range(n):
        for k in range(3):
            crd1[i][k] -= com1[k]
            crd2[i][k] -= com2[k]

    x = matrix(crd1)
    y = matrix(crd2)
    r = transpose(y)*x

    v,s,w_trans = svd(r)
    v = matrix(v)
    w_trans = matrix(w_trans)

    #chiral case
    #small=pow(10,20)
    #if det(v)*det(w_trans)<0:
    #    for i in range(len(s)):
    #        if s[i]<small:
    #            small=copy.deepcopy(s[i])
    #            ismall=i
    #    v[ismall,:]=v[ismall,:]*-1.0

    U = v*w_trans
    rot_y = y*U
    #rmsdv = mat_rmsd(x,rot_y)
    
    crd2_simp = [[0.0,0.0,0.0] for i in range(n)]
    for i in range(n):
        for k in range(3):
            crd2_simp[i][k] = rot_y[i,k] + com1[k]
    return crd2_simp, U, com1

def blosum_score(seq1,seq2,w,report=False):
    score = 0
    for i,aa1 in enumerate(seq1):
        aa2 = seq2[i]
        #score += w[i]*blosum62(aa1,aa2)
        match = blosum62(aa1,aa2)
        if match > 0:
            score += 1.0 #match
    if report:
        print( 'BLOSUM %s %s %.1f'%(seq1,seq2,score) )
    return score*1.0/len(seq1)

class SSclass:
    def __init__(self,pdbid,seq,SS,begin,end,refpos=-1):
        self.pdbid = pdbid
        self.seq = seq
        self.SS = SS
        self.nres = len(seq)
        self.frame = []
        self.cenpos = self.nres/2 
        self.cenres = begin + self.cenpos
        
        self.refpos = self.cenpos 
        if refpos != -1: 
            self.refpos = refpos
            
        self.begin = begin
        self.end = end
        self.reslist = range(begin,end+1)
        self.CAcrds = [[] for k in range(self.nres)]
        self.CBcrds = [[] for k in range(self.nres)]
        self.bbcrds = [{} for k in range(self.nres)]
        self.bbcrds_al = None
        self.axes = [[],[],[]]
        self.iSS = 0 # SSindex (only for SS at protein)
        #self.is_ulr = False
        self.paired_res = {}
        self.hydrophobics = []

    # calculate from CAcrds
    def get_axes(self):
        if self.frame == [] and self.bbcrds != [] and self.CAcrds != []:
            com = self.CAcrds[self.cenpos]
            z = [0.0,0.0,0.0]
            for i,crd in enumerate(self.CAcrds[:-1]):
                crd2 = self.CAcrds[i+1]
                for k in range(3):
                    z[k] += crd2[k]-crd[k]
            normalize(z)

            # x not being used now... trash numbers
            CBcrd = self.CBcrds[self.cenpos]
            x = [CBcrd[k]-com[k] for k in range(3)]
            #if d2(CBcrd,com) > 1.0:
            #    x = [CBcrd[k]-com[k] for k in range(3)]
            #else: #virtual cb
            #    bbcrd = self.bbcrds[self.cenpos]
            #    vNCA = [bbcrd["CA"][k]-bbcrd["N"][k]]
            #    vCCA = [bbcrd["CA"][k]-bbcrd["C"][k]]
            #    xp1 = normalize(cross(vNCA,vCCA))
            #    xp2 = normalize(inproduct(vNCA,vCCA))
            #    c1 = sin(2.0*pi/3.0)
            #    c2 = cos(2.0*pi/3.0)
            #    x = [c1*xp1[k]+c2*xp2[k] for k in range(3)]
            #normalize(x)
            #z = [5.0*z[k]+com[k] for k in range(3)]
            #x = [x[k]+com[k] for k in range(3)]
            self.axes = [com,z,x]
            
    def get_frame(self,update_axes=True,n=5):
        self.frame = [[] for k in range(n)]
        half = n/2

        for i,k in enumerate(range(-half,half+1)):
            self.frame[i] = self.CAcrds[self.refpos+k]
        if update_axes:
            self.get_axes()

    def get_refcrd(self):
        return self.CAcrds[self.refpos]

    def get_sign(self,SS2):
        match = inproduct(self.axes[1],SS2.axes[1])
        if match > 0.5: sgn = 1
        elif match < -0.5: sgn = -1
        else: sgn = 0
        return sgn
            
    def make_subSS(self,i1,i2):
        seq = self.seq[i1:i2+1]
        SS_w = SSclass(self.pdbid,seq,self.SS,self.begin+i1,self.begin+i2)

        SS_w.CAcrds = self.CAcrds[i1:i2+1]
        SS_w.CBcrds = self.CBcrds[i1:i2+1]
        SS_w.bbcrds = self.bbcrds[i1:i2+1]
        SS_w.get_frame()
        return SS_w
        
    def thread_match(self,SSmatch,seqscorecut=0.3):
        # make sure len of SSmatch < len of ULR
        seq1 = self.seq
        seq2 = SSmatch.seq
        n = len(seq2)

        threads = []
        scores = [0.0] #for min(scores)

        # weight more on close-to-refpos
        w_triangular = []
        half = n/2
        for i in range(n):
            shift = abs(SSmatch.refpos-i)
            w_triangular.append(max(0,half-shift))
        wsum = float(sum(w_triangular))
        w_triangular = [w/wsum for w in w_triangular]

        for i in range(len(seq1)-n):
            seq1_p = ''.join([seq1[i+j] for j in range(n)])
            seqscore = blosum_score(seq1_p,seq2,w_triangular)
            #seqscore *= 1.0/n #lower the better -- skip here; replaced by weighting
            scores.append(seqscore)
            if seqscore >= seqscorecut: #more than 30% matched
                threads.append((seqscore,self.begin+i,self.begin+i+n-1))
        threads.sort()
        threads.reverse()
        return threads, min(scores)

    def is_contacting_SS(self,SS2):
        ncontact = 0
        for crd1 in self.CBcrds:
            for crd2 in SS2.CBcrds:
                if d2(crd1,crd2) < 64.0:
                    ncontact += 1
                if ncontact >= 2:
                    return True # 1 is too few
        return False

    # use RMSD...
    def close(self,SS2,take_aligned=True,rmsdcut=1.0):
        if self.nres != SS2.nres: return False

        if take_aligned:
            crd1 = self.CAcrds_al
            crd2 = SS2.CAcrds_al
        else:
            crd1 = self.CAcrds
            crd2 = SS2.CAcrds

        msd = 0.0
        for ires in range(self.nres):
            msd += d2(crd1[ires],crd2[ires])
        msd /= self.nres

        if msd < rmsdcut*rmsdcut:
            return True
        else:
            return False

# simple storage for selected insertions
class ChunkMatch:
    def __init__(self,ulrres,anchorres):
        self.ulrres = ulrres #which ulr thread happens
        self.anchorres = anchorres
        self.bbcrds = None
        self.threads = []
        self.SStype = ''

    def add_threads(self,thread):
        self.threads.append(thread)

class Jump:
    def __init__(self,SSs,reliable=[],pdb=''): 
        self.SSs = SSs
        self.nSS = len(SSs)
        self.pdb = pdb
        self.pdbid = pdb.split('/')[-1][:4]
        self.index = '' # for db
        if reliable != []:
            self.reliable = reliable
        else:
            self.reliable = [True for SS in SSs] #default
        self.strscore = 0.0
        self.seqscore = 0.0
        self.register = []
        self.rmsd_match_cut = 2.0

    def tag(self):
        tagstrs = ['%d-%d'%(SS.begin,SS.end) for SS in self.SSs]
        return ','.join(tagstrs)

    def append_SS(self,SS,reliable=True):
        self.SSs.append(SS)
        self.nSS += 1 
        self.reliable.append(reliable)

    def report_chunk_insertion(self,refJump,imatch,threads,isol,out=None,
                               report_as_pdb=""):
        SS = self.SSs[imatch]
        ULR = refJump.SSs[-1]

        chunkindex = '%6s.%d.%d-%d'%(self.index,isol,SS.begin,SS.end)
        header = 'CHUNK %-17s %s ALIGNEDTO'%(chunkindex,SS.seq)
        for iSS,SSanc in enumerate(refJump.SSs):
            if iSS != imatch:
                header += ' %d-%d'%(SSanc.begin,SSanc.end)

        # split if threadable range is too long
        if ULR.nres > 2*SS.nres:
            i_end1 = ULR.nres/2 + SS.nres/2;
            i_start2 = i_end1 + 2 - SS.nres;
            ulrseq1 = ''.join([ULR.seq[i] for i in range(i_end1+1)])
            ulrseq2 = ''.join([ULR.seq[i] for i in range(i_start2,len(ULR.seq))])
            ULRdefs = [(ULR.begin,i_end1+ULR.begin,ulrseq1),
                       (ULR.begin+i_start2,ULR.end,ulrseq2)]

        else:
            ULRdefs = [(ULR.begin,ULR.end,ULR.seq)]

        # anchoring residue of full ULR
        anc1 = ULR.begin-1
        anc2 = ULR.end+1
        insertion_matches = []
        
        for ulr_begin,ulr_end,ulr_seq in ULRdefs:
            if out != None:
                out.write(header+' THREAD %d %d ANCHOR %d %d\n'%(ulr_begin,ulr_end,anc1,anc2))
            match = ChunkMatch(range(ulr_begin,ulr_end+1),(anc1,anc2))
            match.SStype = SS.SS # E/H as string
            match.bbcrds = SS.bbcrds_al
            
            for (seqscore,begin,end) in threads:
                if begin < ulr_begin or end > ulr_end: continue
                match.add_threads(range(begin,end+1))

                i1 = begin - ulr_begin
                i2 = end - ulr_begin
                seq = ''.join([ulr_seq[i] for i in range(i1,i2+1)])
                
                if out != None:
                    out.write('THREAD %3d %3d          %s %8.3f %8.2f\n'%(begin,end,seq,self.strscore,seqscore))
            if out != None:
                for ires,res in enumerate(SS.reslist):
                    out.write('COORD'+' %8.3f'*12%tuple(SS.bbcrds_al[ires]['N']+SS.bbcrds_al[ires]['CA']+SS.bbcrds_al[ires]['C']+SS.bbcrds_al[ires]['O'])+'\n')
                out.write('END\n')
                    
            insertion_matches.append(match)

        if report_as_pdb != "":
            # report as identity from original DB, not from reference pose
            out2 = file(report_as_pdb,'w')
            pdbform = 'ATOM  %5d  %-2s %4s %1s %3d   %8.3f%8.3f%8.3f  1.00  1.00 %s\n'
            iatm = 0
            # report anchor
            for iSS,SS2 in enumerate(self.SSs):
                if iSS == imatch: continue
                if SS2.bbcrds_al == None:
                    #print "Warning: skip reporting "
                    continue

                for ires,res in enumerate(SS2.reslist):
                    aa3 = aa1toaa3(SS2.seq[ires])
                    for atm in ['N','CA','C','O']:#SS.bbcrds[ires]:
                        iatm += 1
                        acrd = SS2.bbcrds_al[ires][atm]
                        out2.write(pdbform%(iatm,atm,aa3,'A',res,
                                            acrd[0],acrd[1],acrd[2],atm[0]))
            # report ULR
            for ires,res in enumerate(SS.reslist):
                aa3 = aa1toaa3(SS.seq[ires])
                for atm in ['N','CA','C','O']:#SS.bbcrds[ires]:
                    iatm += 1
                    acrd = SS.bbcrds_al[ires][atm]
                    out2.write(pdbform%(iatm,atm,aa3,'B',res,
                                       acrd[0],acrd[1],acrd[2],atm[0]))
            out2.close()
            
        return insertion_matches

    def get_bbcrd_from_pdb(self):
        if self.pdb == '': return
        bbcrds = pdb2crd(self.pdb,'mc')
        CAcrds = pdb2crd(self.pdb,'CA')
        CBcrds = pdb2crd(self.pdb,'CB')
        for iSS,SS in enumerate(self.SSs):
            self.SSs[iSS].bbcrds = [[] for res in SS.reslist]
            self.SSs[iSS].CAcrds = [[] for res in SS.reslist]
            self.SSs[iSS].CBcrds = [[] for res in SS.reslist]
            for ires,res in enumerate(SS.reslist):
                self.SSs[iSS].bbcrds[ires] = bbcrds[res]
                self.SSs[iSS].CAcrds[ires] = CAcrds[res]
                self.SSs[iSS].CBcrds[ires] = CBcrds[res]

            self.SSs[iSS].get_frame()
        
    def is_SSconsistent(self,QJump,segorder):
        for i,SS1 in enumerate(self.SSs):
            j = segorder[i]
            SS2 = QJump.SSs[j]
            #print i, self.reliable[i], QJump.reliable[j], SS1.SS, SS2.SS
            if not (self.reliable[i] and QJump.reliable[j]): continue
            if SS1.SS != SS2.SS: return False
        return True

    def walk_register(self,SS1,SS2,seedpair,extendable,sgn,fill_gap=False):
        regs = []
        
        nhalf1 = SS1.nres/2
        nhalf2 = SS2.nres/2
        for k in range(-nhalf1,nhalf1+1):
            i1 = seedpair[0]+k
            i2 = seedpair[1]+sgn*k
            if i1 < 0 or i2 < 0: continue
            if (i1,i2) in extendable:
                regs.append((i1,i2))

        # reverse direction
        #for k in range(-nhalf1,nhalf1+1):
        #    if k == 0: continue #skip center
        #    i1 = seedpair[0]+k
        #    i2 = seedpair[1]-sgn*k
        #    print i1,i2
        #    if i1 < 0 or i2 < 0: continue
        #    if (i1,i2) in extendable:
        #        regs.append((i1,i2))

        regs.sort()

        if len(regs) > 1 and fill_gap:
            # fill missing gap
            for ireg,reg1 in enumerate(regs[:-1]):
                reg2 = regs[ireg+1]
                for i,i1 in enumerate(range(reg1[0],reg2[0])):
                    i2 = reg1[1]+sgn*i
                    if (i1,i2) not in regs:
                        regs.append((i1,i2))
                
        return regs
    
    # find direct contacting (e.g. sheet-pairing) residues
    # currently not using seed info
    def find_registering_respairs(self, seed={}):
        self.register = [{} for i in range(self.nSS)]
        
        for iSS,SS1 in enumerate(self.SSs):
            for jSS,SS2 in enumerate(self.SSs):
                if jSS <= iSS: continue

                # check the axes of SS; skip if not parallel nor antiparallel
                sgn = SS1.get_sign(SS2)
                
                cen1 = SS1.cenpos
                cen2 = SS2.cenpos
                
                # first get seed
                EE = (SS1.SS == 'E' and SS2.SS == 'E')

                sortable = []
                for i1,res1 in enumerate(SS1.reslist):
                    for i2,res2 in enumerate(SS2.reslist):
                        if EE:
                            #crd1 = SS1.CBcrds[i1]
                            #crd2 = SS2.CBcrds[i2]
                            crd1 = SS1.CAcrds[i1]
                            crd2 = SS2.CAcrds[i2]
                            dis2 = d2(crd1,crd2)
                            if dis2 < 49.0:
                                sortable.append((abs(cen1-i1)+abs(cen2-i2),(i1,i2)))
                        else: #HH
                            crd1 = SS1.CAcrds[i1]
                            crd2 = SS2.CAcrds[i2]
                            dis2 = d2(crd1,crd2)
                            if dis2 < 49.0:
                                sortable.append((abs(cen1-i1)+abs(cen2-i2),(i1,i2)))
                                
                if len(sortable) == 0: continue
                sortable.sort()
                self.alt_seeds = [comp[1] for comp in sortable]

                if EE: #take simple
                    seedpair = sortable[0][1]
                    regs_with_max_match = self.walk_register(SS1,SS2,seedpair,self.alt_seeds,sgn)
                    # extend
                    
                    
                else: # HH
                    sort_by_matches = []
                    # try alternatives...
                    for seedtry in self.alt_seeds:#
                        regs = self.walk_register(SS1,SS2,seedtry,self.alt_seeds,sgn,fill_gap=True)
                    sort_by_matches.append([len(regs),seedtry,regs])
                    
                    sort_by_matches.sort()
                    regs_with_max_match = sort_by_matches[-1][2]
                    seedpair = sort_by_matches[-1][1]
                
                self.register[iSS][jSS] = regs_with_max_match
                self.register[jSS][iSS] = [(comp[1],comp[0]) for comp in regs_with_max_match] #inverted order

                #print 'seed: ', seedpair, SS1.reslist[seedpair[0]], SS2.reslist[seedpair[1]]

    def split_to_windows(self,wsize,append_reverse=False,multiple_regs=False,
                         skip_if_strand_paired_in_both_direcs=True,
                         include_hydrophobic=0):
        jump_windows = []

        if len(self.SSs) == 1:
            SS = self.SSs[0]
            for i in range(SS.nres-wsize+1):
                i1 = i
                i2 = i+wsize-1
                usable = True
                # skip if hydrophobic res not included as specified
                if include_hydrophobic:
                    nHP = 0
                    for ires in range(i1,i2+1):
                        resno = SS.begin + ires
                        if resno in SS.hydrophobics:
                            nHP += 1
                    if nHP < include_hydrophobic:
                        usable = False
                
                # skip if the strand already paired in both directions
                if SS.SS == 'E' and skip_if_strand_paired_in_both_direcs:
                    for ires in range(i1,i2+1):
                        resno = SS.begin + ires
                        if resno in SS.paired_res and len(SS.paired_res[resno]) >= 2:
                            usable = False
                            print( '---  chunk %d-%d not usable as a register because of both-paired res %d'%(SS.begin+i1,SS.begin+i2,resno) )
                            break
                if not usable: continue
                    
                SS_w = SS.make_subSS(i1,i2)
                jump_windows.append(Jump([SS_w]))
        
        elif len(self.SSs) == 2:
            half = wsize/2
            iSS1,iSS2 = 0,1 #HACK


            if iSS2 not in self.register[iSS1]: return []
            regs_ij = self.register[iSS1][iSS2]

            if len(regs_ij) < wsize: return []

            for i in range(self.SSs[iSS1].nres-wsize+1):
                SS1 = self.SSs[iSS1]
                SS2 = self.SSs[iSS2]
                sgn = SS1.get_sign(SS2)

                shift1 = (i, i+wsize-1) #new ibegin/iend of subSS1
                SS1_w = SS1.make_subSS(shift1[0],shift1[1])

                match = []
                for (i1,i2) in regs_ij: # get range of registers within window 
                    if i1 >= i and i1 <= i+wsize-1: match.append((i1,i2))
                # take consistent subset?
                if len(match) < wsize: continue

                #new ibegin/iend of subSS2
                shift2 = (min(match[0][1],match[-1][1]),max(match[0][1],match[-1][1])) 

                if multiple_regs: ## HH jump
                    for offset in range(-4,5): #[-4,-3,-2,-1,0,1,2,3,4]:
                        (i1,i2) = (shift2[0]+offset,shift2[1]+offset)
                        if i2-i1 < MINRES_FOR_FRAME-1: continue
                        if i1 < 0 or i2 >= SS2.nres: continue

                        SS2_w = SS2.make_subSS(i1,i2)
                        jump = Jump([SS1_w,SS2_w],reliable=[True,True],pdb=self.pdb)
                        jump_windows.append(jump)
                else:
                    if shift2[1]-shift2[0] < MINRES_FOR_FRAME-1: continue
                    SS2_w = SS2.make_subSS(shift2[0],shift2[1])
                    jump = Jump([SS1_w,SS2_w],reliable=[True,True],pdb=self.pdb)
                    jump_windows.append(jump)
            
        else: # nSS > 2
            sys.exit('jump.split_to_windows currently does not work nSS != 2')
                
        return jump_windows

    def get_possible_permutations(self,anchorSSs=[]):
        if self.nSS == 2:
            combs = [[0,1],[1,0]]
        elif self.nSS == 3:
            combs= [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]

        for comb in copy.copy(combs):
            not_thisone = False
            for iSS in anchorSSs:
                if iSS == comb[-1]:
                    not_thisone = True
                    break
            if not_thisone:
                combs.remove(comb)
        return combs
    
    def match_jump(self,QJump,
                   anchorSSs=[]):
        possible_segorders = self.get_possible_permutations(anchorSSs)

        crd_r = [] #reference frame; always n-th is unreliable
        for i in range(self.nSS-1):
            crd_r += self.SSs[i].frame

        # Search through different permutations
        # in segorder [1:n-1] are reference, n-th is 
        sortable = []
        for iorder,segorder in enumerate(possible_segorders):
            #Filter by SStype
            if not self.is_SSconsistent(QJump,segorder): continue

            # Get partial structures for alignment from query jump
            crd_q = [] #apply permutation from segorder
            for i in segorder[:-1]:
                crd_q += QJump.SSs[i].frame

            if len(crd_r) != len(crd_q):
                sys.exit('Inconsistent length for match: %d vs %d'%(len(crd_r), len(crd_q)) )

            crd_q_al, U, t = Kabsch(crd_r,copy.deepcopy(crd_q))
            rmsd_anchor = rmsd_simple(crd_r,crd_q_al)
            #sortable.append((rmsd_anchor,imatch,U,t))
            sortable.append((rmsd_anchor,segorder,U,t))

        solutions = []
        
        if len(sortable) <= 1: return solutions
        sortable.sort()
        
        for isol,comp in enumerate(sortable):
            rmsd = comp[0]
            #print rmsd
            if rmsd > self.rmsd_match_cut: break

            segorder = comp[1]
            U = comp[2]

            SSs_reordered = [copy.deepcopy(QJump.SSs[i]) for i in segorder]
            QJump_al = Jump(SSs_reordered)
            QJump_al.index = QJump.index

            QJump_al.transrot_bbcrd(U,t,moving_iSS=[self.nSS-1])
            QJump_al.strscore = rmsd

            solutions.append(QJump_al)

        return solutions

    def transrot_bbcrd(self,U,t,moving_iSS):
        iSSs = range(len(self.SSs))
        
        # first get com
        com = [0.0,0.0,0.0]
        n = 0
        for iSS in iSSs: 
            SS = self.SSs[iSS]
            if iSS in moving_iSS: continue
            for ires,resCAcrd in enumerate(SS.frame):
                acrd = resCAcrd
                for k in range(3):
                    com[k] += acrd[k]
                n += 1
        com = [com[k]/n for k in range(3)]

        for iSS in iSSs: 
            SS = self.SSs[iSS]
            self.SSs[iSS].bbcrds_al = [{} for res in SS.bbcrds]
            self.SSs[iSS].CAcrds_al = [[] for res in SS.CAcrds]
            for ires,resbbcrd in enumerate(SS.bbcrds):
                for atm in resbbcrd:
                    acrd = [resbbcrd[atm][k]-com[k] for k in range(3)]

                    acrd_al = [0.0,0.0,0.0]
                    for k in range(3):
                        acrd_al[k] = U[0,k]*acrd[0] + U[1,k]*acrd[1] + U[2,k]*acrd[2] + t[k]
                    self.SSs[iSS].bbcrds_al[ires][atm] = acrd_al

                    if atm == 'CA':
                        self.SSs[iSS].CAcrds_al[ires] = acrd_al

class JumpDBtype:
    def __init__(self,prefix,out=sys.stdout):
        self.prefix = prefix
        self.db = []
        self.report_pdb = False
        self.duplication_rmsd_cut = 5.0
        self.out = out
        self.chunkmatches = []
        
    def scan_region(self,RefJump,protein,outprefix='',report_as_a_file=True):
        ULR = RefJump.SSs[-1]
        seqpos = ULR.reslist[0] # starting seqpos

        #print "Searching through %d possible jumps in DB"%(len(self.db))
        jump_and_thread_solutions = []
        ndupl = 0
        nclash = 0
        nmatches = 0
        self.chunkmatches = []

        for ijump,jump in enumerate(self.db):
            solutions = RefJump.match_jump(jump)

            nmatches += len(solutions)
            if solutions != []:
                for isol,jump_al in enumerate(solutions):
                    # filter by clash
                    if protein.SS_is_clashing(jump_al.SSs[-1],clash_cut=CLASHCUT):
                        nclash += 1
                        continue
                    
                    threads,bestmatch = ULR.thread_match(jump_al.SSs[-1],seqscorecut=SEQSCORECUT)
                    jump_and_thread_solutions.append((jump_al,threads,isol))
                    
        selectedSS = []
        for i,(jump_al,threads,isol) in enumerate(jump_and_thread_solutions):
            SSreplaced = jump_al.SSs[-1]

            # check if similar one already placed 
            duplicate = False
            for SS in selectedSS:
                if SSreplaced.close(SS,rmsdcut=self.duplication_rmsd_cut):
                    duplicate = True
                    break
            if duplicate:
                ndupl += 1
            else:
                if len(threads) == 0: continue
                
                if self.report_pdb:
                    pdbname = 'match.%s.%s.%d.pdb'%(outprefix,jump_al.index,isol)
                else:
                    pdbname = ''

                outfile=None
                if report_as_a_file:
                    outfile = self.out
                    
                self.chunkmatches += jump_al.report_chunk_insertion(RefJump,RefJump.nSS-1, #always sorted as final
                                                              threads, isol, out=outfile,
                                                              report_as_pdb=pdbname
                )
                selectedSS.append(SSreplaced)

        print( "--  Match %s: Total %3d jump solutions found (%d/%d filtered from %d by clash/struct-similarity"%(outprefix,len(selected),nclash,ndupl,nmatches) )


    def scan_through_generic(self,protein,jumps,ulrs,nanchor,include_hydrophobic=0):
        if nanchor > 2:
            print( "not supporting nachor more than 2" )
            return 
        
        for iulr,ulr in enumerate(ulrs):
            print( "\nULR %d-%d, scan through %d jumps:"%(ulr.begin,ulr.end,len(jumps)) )
            for ipair,jump in enumerate(jumps):
                jumpdef = ",".join(["%d-%d"%(SS.begin,SS.end) for SS in jump.SSs]) 

                if not protein.ulr_placeable_at_jump(jump,ulr.reslist):
                    print( "- NOT placeable at Jump %s"%(jumpdef) )
                    continue

                print( "- placeable at Jump %s"%(jumpdef) )

                #build up as registering pair of 5mers
                jump_windows = jump.split_to_windows(5,
                                                     multiple_regs=(self.prefix in ['HH']), #CHECK!!
                                                     #multiple_regs=True,
                                                     include_hydrophobic=include_hydrophobic)
                
                print( '-  Searching %d possible registers for jump %d.'%(len(jump_windows),ipair) )
                
                for i_w,jump_w in enumerate(jump_windows):

                    prefix = ",".join(["%d.%d"%(SS.begin,SS.end) for SS in jump_w.SSs]) 
                    prefix += "-%d.%d"%(ulr.begin,ulr.end)
                    
                    jump_w.append_SS(ulr,reliable=False)
                    self.scan_region(jump_w,protein,outprefix=prefix)

    def scan_through_unpaired_strands(self,protein,nanchor):
        unpaired_strands_one, unpaired_strands_two = protein.find_strand_pairs(find_unpaired=True)
        
        # can be customized later...
        ulrs = protein.ULRs
        for iulr,ulr in enumerate(ulrs):
            print( 'ULR             %2d: %d-%d'%(iulr,ulr.reslist[0],ulr.reslist[-1]) )
        
        if nanchor == 1:
            for istrand,strand in enumerate(unpaired_strands_one):
                print( 'unpaired strand %2d: %d-%d'%(istrand,strand.begin,strand.end) )

            unpaired_strands_one_as_jumps = []
            for strand in unpaired_strands_one:
                unpaired_strands_one_as_jumps.append(Jump([strand]))
            self.scan_through_generic(protein,unpaired_strands_one_as_jumps,ulrs,1)
            
            #self.scan_through_generic_legacy(protein,unpaired_strands_one,ulrs,1)

        elif nanchor == 2:
            for ijump,jump in enumerate(unpaired_strands_two):
                jump.rmsd_match_cut = 1.0 # works here?
                SS1 = jump.SSs[0]
                SS2 = jump.SSs[1]
                print( 'unpaired jump %2d: %d-%d %d-%d'%(ijump,SS1.begin,SS1.end,
                                                         SS2.begin,SS2.end))
            self.scan_through_generic(protein,unpaired_strands_two,ulrs,2)

    def scan_through_exposed_helix(self,protein): #nanchor=1
        exposed_one = protein.find_exposed_helix()
        
        # can be customized later...
        ulrs = protein.ULRs
        for iulr,ulr in enumerate(ulrs):
            print( 'ULR             %2d: %d-%d'%(iulr,ulr.reslist[0],ulr.reslist[-1]) )
        
        for ihelix,helix in enumerate(exposed_one):
            print( 'exposed helix %2d: %d-%d'%(ihelix,helix.begin,helix.end) )

        for helix in exposed_one:
            #exposed_helix_as_jumps.append()
            self.scan_through_generic(protein,[Jump([helix])],ulrs,1,
                                      include_hydrophobic=2)
            
    def scan_through_exposed_SSpair(self,protein,nanchor,
                                    SStype='H'):
        if SStype == 'H':
            exposed_two = protein.find_exposed_helices()
        elif SStype == 'E':
            exposed_two = protein.find_exposed_strands()
        else:
            sys.exit('scan_through_exposed_SS: No such SStype of %s'%SStype)

        # can be customized later...
        print( "Reporting ULRs and Jump" )
        ulrs = protein.ULRs
        for iulr,ulr in enumerate(protein.ULRs):
            print( 'ULR             %2d: %d-%d'%(iulr,ulr.reslist[0],ulr.reslist[-1]) )

        print( "Reporting possible jumps:" )
        for ijump,jump in enumerate(exposed_two):
            SS1 = jump.SSs[0]
            SS2 = jump.SSs[1]
            print( 'Exposed jump %2d: %d-%d %d-%d'%(ijump,SS1.begin,SS1.end,
                                                    SS2.begin,SS2.end))
        #exposed_two = [exposed_two[2]] #HACK
        self.scan_through_generic(protein,exposed_two,ulrs,nanchor)
           
                    
class JumpDB:
    def __init__(self,path,maxlibrank=100000):
        self.dbtype = {}
        self.maxlibrank = maxlibrank
        dbfiles = glob.glob('%s/*H.chunklib'%path)+glob.glob('%s/*E.chunklib'%path)
        for dbf in dbfiles:
            prefix = dbf.split('/')[-1].split('.')[0]
            self.read_dbfile(dbf,prefix)

    def read_dbfile(self,dbfile,prefix):
        self.dbtype[prefix] = JumpDBtype(prefix)

        jumps = []
        iSSprv = -1
        for l in file(dbfile):
            #CHUNK tag N SS1 begin1 end1 seq1; SS2 begin2 end2 seq2;
            if l.startswith('CHUNK'):
                strs = l[:-2].split(';')

                words = strs[0].split()
                tag = words[1]
                rank = int(tag)
                if rank > self.maxlibrank: continue
                nSS = int(words[2])

                SSs = []
                SSstrs = strs[1:]
                for iSS in range(nSS):
                    words = SSstrs[iSS].split()
                    SS = words[0]
                    begin = int(words[1])
                    end = int(words[2])
                    seq = words[3]
                    refpos = int(words[4])-begin
                    SSseg = SSclass(tag,seq,SS,begin,end,refpos=refpos)
                    SSs.append(SSseg)
                    
                jump = Jump(SSs,pdb="")
                jump.index = tag
                jumps.append(jump)
                #bbcrds = [[] for k in range(nSS)]
                    
            elif l.startswith('COORD'):
                if rank > self.maxlibrank: continue
                words = l[:-1].split()
                #COORD iseg res Nx Ny Nz CAx CAy CAz Cx Cy Cz Ox Oy Oz
                iSS = int(words[1])
                SS = jumps[-1].SSs[iSS]
                if iSS != iSSprv:
                    ires = 0
                    
                bbcrd = {}
                bbcrd['N']  = [float(words[3]),float(words[4]), float(words[5])]
                bbcrd['CA'] = [float(words[6]),float(words[7]), float(words[8])]
                bbcrd['C']  = [float(words[9]),float(words[10]),float(words[11])]
                bbcrd['O']  = [float(words[12]),float(words[13]),float(words[14])]
                SS.bbcrds[ires] = bbcrd
                SS.CAcrds[ires] = bbcrd['CA']
                
                iSSprv = iSS
                ires += 1
                
                if ires == SS.nres:
                    SS.get_frame()

        self.dbtype[prefix].db = jumps

    def set_report_pdb(self,setting):
        for prefix in self.dbtype:
            self.dbtype[prefix].report_pdb = setting

