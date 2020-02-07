import sys, copy, os
import numpy as np

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,'%s/../basic'%SCRIPTDIR)
from PoseInfo import SSclass

sys.path.insert(0,'%s/../../utils'%SCRIPTDIR)
from myutils import write_as_pdb

#===========================================================
# misc funcs.
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

def Kabsch(crd1_in,crd2_in,w=[]):
    from scipy.linalg import svd
    from scipy import matrix, array, transpose
    
    n = len(crd1_in)
    if w == []: w = np.array([1.0 for i in range(n)])

    com1 = np.mean(crd1_in,axis=0)
    com2 = np.mean(crd2_in,axis=0)
    crd1 = crd1_in - com1
    crd2 = crd2_in - com2

    x = matrix(crd1)
    y = matrix(crd2)
    r = transpose(y)*x

    v,s,w_trans = svd(r)
    v = matrix(v)
    w_trans = matrix(w_trans)

    U = v*w_trans
    rot_y = y*U
    
    crd2_simp = np.zeros((n,3)) 
    for i in range(n): crd2_simp[i] = rot_y[i] + com1
    return crd2_simp, np.array(U), com1

#===========================================================

# Stores info about solution -- all the static info lives in PoseInfo or TERM.py
# more details: permutation, TermIndex, score, U, t, threads
class MatchSolution:
    def __init__(self,U,t,
                 SSs_term,anchor_com,
                 index,
                 anchorres,
                 aln_score=0.0,
                 tag="" ):
        self.U = U
        self.t = t
        self.tag = tag

        self.SSs_term = SSs_term
        self.anchor_com = anchor_com
        self.anchorres = anchorres
        #self.SSorder = SSorder #permutation in TERM. e.g, [0,2,1]
        
        self.index = index # Term index
        self.threads = [] #(score,begin,end)
        self.aln_score = aln_score

    def transrot_bbcrd(self):
        for iSS,SS in enumerate(self.SSs_term): 
            acrds = SS.bbcrds - self.anchor_com
            acrd_al = np.matmul(acrds,self.U) + self.t
            self.SSs_term[iSS].bbcrds_al = acrd_al
            self.SSs_term[iSS].CAcrds_al = acrd_al[:,1]

    def write_as_pdb(self,outpdb):
        crds = np.zeros((0,5,3))
        for SS in self.SSs_term: 
            crds = np.concatenate((crds,SS.bbcrds_al))
        write_as_pdb(outpdb,crds)
        
    # Used in Matcher... moved from SSclass
    def find_threads(self,
                     estogram,SSpred,
                     seqscorecut=0.3):
        
        # make sure len of SSmatch < len of ULR
        seq1 = ULR
        seq2 = SSmatch.seq
        n = len(seq2)

        # weight more on close-to-refpos
        half = int(n/2)
        w = np.array([max(0,half-abs(SSmatch.refpos-i)) for i in range(n)])
        w /= np.sum(w)

        score_SS  = 0.0 # match to SSpred
        score_d   = 0.0 # match to estogram
        score_seq = 0.0 # match to blosum
        
        threads = []
        for i in range(len(seq1)-n):
            seq1_p = ''.join([seq1[i+j] for j in range(n)])
            score_seq = blosum_score(seq1_p,seq2,w_triangular)

            d = distance(crds_insertd[res1],crds_noulr[res2])
            score_d += lookup(estogram[res1][res2],self.dbins,d)

            score_SS += SS3prob[res]

            score = -(score_dist + score_SS + score_seq)
            scores.append(seqscore)
            threads.append((score,ULR.begin+i,ULR.begin+i+n-1))
        threads.sort()
        self.threads = threads

class TERMClass:
    def __init__(self,SSs,tag=''): 
        self.SSs = SSs
        self.nSS = len(SSs)
        self.tag = tag
        self.index = '' # for db

    def tag(self):
        tagstrs = ['%d-%d'%(SS.begin,SS.end) for SS in self.SSs]
        return ','.join(tagstrs)

    def find_solutions(self,query,rmsd_match_cut):
        # query: TERM, reference: Pose 
        # First get self permutations
        possible_segorders = self.get_possible_permutations()

        # List of reference frame from Pose: [x1,y1,z1,x2,y2,z2]
        crd_r = np.zeros((0,3))
        for SS in query[:-1]: crd_r = np.concatenate((crd_r,SS.frame))

        # Search through different permutations
        # in segorder [:n-1] are reference, the final is ULR-SS
        sortable = []
        for iorder,segorder in enumerate(possible_segorders):
            if not self.is_SSconsistent(query,segorder): continue
            
            # List of reference frame from query: [x1,y1,z1,x2,y2,z2]
            crd_q = np.zeros((0,3))
            for i in segorder[:-1]: crd_q = np.concatenate((crd_q,self.SSs[i].frame))
            
            if len(crd_r) != len(crd_q): return []

            crd_q_al, U, t = Kabsch( crd_r, crd_q ) 

            dcrd = crd_r-crd_q_al
            rmsd_anchor = np.sqrt(np.mean(dcrd*dcrd)*3)
            anccom = np.mean(crd_q,axis=0) #original anchor position in TERM
            
            #sort by similarity of anchor
            sortable.append((rmsd_anchor,segorder,U,t,anccom))

        solutions = []
        
        if len(sortable) <= 1: return solutions
        sortable.sort()
        
        ancres_in_pose = [SS.cenres for SS in query] #static
        for isol,(rmsd,segorder,U,t,anccom) in enumerate(sortable):
            if rmsd > rmsd_match_cut: break 

            #always ulr goes last
            SSs_reordered = [copy.deepcopy(self.SSs[i]) for i in segorder] 
            match = MatchSolution( U, t,
                                   SSs_reordered, anccom,
                                   self.index,
                                   ancres_in_pose,
                                   aln_score=rmsd,
                                   tag="%s.%02d"%(self.index,isol))
            solutions.append( match ) # bbcrd_al not calculated here
        return solutions

    def is_SSconsistent(self,query,segorder):
        if len(query) != len(segorder): return False #why this happening
        for i,SS1 in enumerate(self.SSs):
            j = segorder[i]
            SS2 = query[j]
            if SS1.SStype != SS2.SStype: return False
        return True

    def get_possible_permutations(self):
        if self.nSS == 2:
            combs = [[0,1],[1,0]]
        elif self.nSS == 3:
            combs= [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        return combs
    
class TERMDB:
    def __init__(self,opt,SStype):
        self.SStype = SStype
        self.db = [] #list of TERMClass
        
        dbf = '%s/%s.chunklib'%(opt.config['TERM_DBPATH'],SStype)
        maxlibrank = opt.config['MAXLIBRANK'][SStype]
        self.read_dbfile(dbf,maxlibrank)

        #store for future usage
        self.rmsd_anchor_cut = opt.config["RMSD_MATCH_CUT"]
        
    def read_dbfile(self,dbfile,maxlibrank):
        self.db = []
        iSSprv = -1
        for l in open(dbfile):
            #CHUNK tag N SS1 begin1 end1 seq1; SS2 begin2 end2 seq2;
            if l.startswith('CHUNK'):
                strs = l[:-2].split(';')

                words = strs[0].split()
                tag = words[1]
                rank = int(tag)
                if rank > maxlibrank: continue
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
                    SSseg = SSclass(SS,begin,end,refpos=refpos)
                    SSs.append(SSseg)
                    
                term = TERMClass(SSs,tag="")
                term.index = rank #tag
                self.db.append(term)
                    
            elif l.startswith('COORD'):
                if rank > maxlibrank: continue
                words = l[:-1].split()
                #COORD iseg res Nx Ny Nz CAx CAy CAz Cx Cy Cz Ox Oy Oz
                iSS = int(words[1])
                SS = self.db[-1].SSs[iSS]
                if iSS != iSSprv: ires = 0
                    
                bbcrd = np.array([[float(words[3]),float(words[4]), float(words[5])],
                                  [float(words[6]),float(words[7]), float(words[8])],
                                  [float(words[9]),float(words[10]),float(words[11])],
                                  [float(words[12]),float(words[13]),float(words[14])],
                                  [0.0,0.0,0.0]] #CB -- unused
                )
                SS.bbcrds[ires] = bbcrd
                SS.CAcrds[ires] = bbcrd[1,:]
                
                iSSprv = iSS
                ires += 1
                
                if ires == SS.nres:
                    SS.get_frame()

        # validate
        #for i,term in enumerate(self.db):
        #    for j,SS in enumerate(term.SSs):
        #        print( self.SStype+ " TERM %s %d/%d"%(term.index,i,j), SS.nres, len(SS.frame) )
                
