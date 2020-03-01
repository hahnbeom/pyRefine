import numpy as np

## "jump": Term or MatchSolution class
def distance(crd1,crd2):
    dv = crd1-crd2
    return np.sqrt(np.sum(dv*dv))

# Inherits prv. "JumpDBtype"
class Matcher:
    def __init__(self,db,opt,prefix,debug=False,rmsdcut_from_init=99.9):
        self.prefix = prefix
        self.db = db #TermDB, for instance
        self.size = len(prefix) #HH:2, for instance
        self.report_pdb = debug #should move to opt
        self.duplication_rmsd_cut = opt.config['DUPLICATION_RMSD'] #5.0
        self.solutions = []
        self.rmsd_anchor_cut = opt.config['RMSD_ANCHOR_CUT'] #2.0
        self.clash_cut = opt.config['MAX_CLASH_COUNT']
        self.rmsdcut_from_init = rmsdcut_from_init #pass as arg because its not constant 

    # previously called "scan_through_generic"
    def place_ULR_at_anchors(self,poseinfo,anchors,ulr_t, #ulr_t should be defined as SSclass
                             include_hydrophobic=0):
        self.solutions = []
        for i,anchor in enumerate(anchors):
            if not self.is_ulr_placeable_at_anchor(poseinfo,anchor,
                                                   ulr_t.reslist):
                continue
            self.solutions += self.find_matches_through_DB(poseinfo,anchor,ulr_t)

        # need "self".solutions?
        return self.solutions

    def is_ulr_placeable_at_anchor(self,poseinfo,anchor,ulrres):
        #first check if ULRseq overlapps with jump -- necessary?
        for SS in anchor:
            for res in ulrres:
                if abs(SS.begin-res) < 3 or abs(SS.end-res) < 3:
                    return True
        
        anccrd1 = []
        anccrd2 = []
        dperres = 3.0
        if ulrres[0]-poseinfo.reslist[0] > 3: #ulr not Nterm
            anccrd1 = poseinfo.CAcrds[ulrres[0]]
        if poseinfo.reslist[-1]-ulrres[-1] > 3: #ulr not Cterm
            anccrd2 = poseinfo.CAcrds[ulrres[-1]]

        anchorcom = np.mean(np.array([SS.get_refcrd() for SS in anchor]),axis=0)

        ## better estimation than nres*3.0 ang (takes into account of folding...)??
        if anccrd1 != [] and anccrd2 != []: #ulr as loop
            half = len(ulrres)/2
            d_jump_to_ulranc1 = distance(anchorcom,anccrd1)
            d_jump_to_ulranc2 = distance(anchorcom,anccrd2)

            if d_jump_to_ulranc1 > dperres*half and d_jump_to_ulranc2 > dperres*half:
                return False
            else:
                return True
            
        else: # terminus case
            if anccrd1 != []: anccrd = anccrd1
            else: anccrd = anccrd2
            d_jump_to_ulranc = distance(anchorcom,anccrd)
            nres_linker = len(ulrres)-3 # rough estimation of max length
            if d_jump_to_ulranc > dperres*nres_linker:
                return False
            else:
                return True

    ## Finds non-redundant, non-clashing term libs + threading options that are closable
    def find_matches_through_DB(self,poseinfo,anchor,ulr_t,
                                outprefix='',report_as_a_file=True):
        seqpos = ulr_t.reslist[0] # starting seqpos

        ### Search against termDB
        query = anchor + [ulr_t]
        solutions = []
        for term in self.db.db:
            solutions += term.find_solutions(query,self.rmsd_anchor_cut)
            #for i,match in enumerate(solutions):
            #    match.write_as_pdb("match.%d."%i+term.tag+"pdb")

        if solutions == []:
            return solutions
            
        #print("GOT %d solutions"%len(solutions))
        anchortag = "-".join(["%03d"%(SS.cenres) for SS in query])
        
        ### Filter
        filtered = []
        nmismatch,nclash,nredundant,nclosure,nthreads = (0,0,0,0,0)
        for i,term in enumerate(solutions):
            # tag anchor
            term.tag = "anc%s."%anchortag+term.tag
            
            # first get aligned crds
            term.transrot_bbcrd()
            #term.write_as_pdb("match.%d."%i+term.tag+"pdb")
            
            # match SS.reslist to pose
            stat = term.redefine_resrange(query,poseinfo.reslist,
                                          ulr=[len(anchor)]) #let ULR undefined

            if not stat:
                #print("%d/%d mismatch"%(i+1,len(solutions)))
                nmismatch += 1
                continue # filter if mapping fails
            
            ULRSS = term.SSs[-1]
            # 1. Filter by Clash
            if poseinfo.SS_is_clashing(ULRSS,clash_cut=self.clash_cut):
                #term.write_as_pdb("clash.%s.pdb"%term.tag) #debugging
                nclash  += 1
                continue

            # 2. Filter by redundancy against stored ones (before threading)
            if self.is_redundant_frame(ULRSS,[f.SSs_term[-1] for f in filtered],
                                       rmsdcut=self.duplication_rmsd_cut): #non-superposition
                #term.write_as_pdb("redundant.%s.pdb"%term.tag) #debugging
                nredundant += 1
                continue

            # 3. Get all possible non-redundant threading options regarding closure condition
            threads = self.find_threads_closable(poseinfo,ULRSS)
            
            if threads == []:
                nclosure += 1
                continue #nothing can be closed

            term.threads = threads
            nthreads += len(threads)
            #term.write_as_pdb("legit.%s.pdb"%term.tag) #debugging
            filtered.append(term)

        l = " - filtered %d/%d/%d/%d by mismatch/clash/redundancy/closure: %d->%d (%d threads)"
        print(l%(nmismatch,nclash,nredundant,nclosure,len(solutions),len(filtered),nthreads))
        return filtered

    # rmsd without superposition
    def is_redundant_frame(self,SS1,SSlist,rmsdcut=2.0):
        for SS2 in SSlist:
            if SS1.close(SS2): return True
        return False

    # Check closability + optional scoring to get threadable ones
    #def score_threads_from_info(self,poseinfo,
    #                              estogram=None, SSpred=None):
        

    # Here just for relevance of its function
    def find_threads_closable(self,poseinfo,SS):
        threads = []

        # first find stem regions
        stemN,stemC = (SS.reslist[0],SS.reslist[-1])
        while True:
            if stemN-1 < 0 or not poseinfo.extended_mask[stemN-2]: break #0-index
            #if stemN-1 < 0 or poseinfo.res_in_SSseg(stemN-1): break
            stemN -= 1
            
        while True:
            if stemC >= len(poseinfo.SS3_naive) or not poseinfo.extended_mask[stemC]: break
            #if stemC >= len(poseinfo.SS3_naive) or poseinfo.res_in_SSseg(stemC+1): break
            stemC += 1
        #print("extend stem from %d/%d to %d/%d"%(SS.reslist[0],SS.reslist[-1],stemN,stemC))
        
        if stemN > 1:
            xyzN_stem = poseinfo.CAcrds[stemN-1] #0-index
            dN2 = np.sum((xyzN_stem-SS.CAcrds_al[0])*(xyzN_stem-SS.CAcrds_al[0]))
        else:
            dN2 = -1 #N-term
            
        if stemC <= poseinfo.nres:
            xyzC_stem = poseinfo.CAcrds[stemC-1] #0-index
            dC2 = np.sum((xyzC_stem-SS.CAcrds_al[-1])*(xyzC_stem-SS.CAcrds_al[-1]))
        else:
            dC2 = -1 #C-term
        
        #scan through cenpos 
        half = int(len(SS.frame)/2)
        for cenpos in range(half,SS.nres-half):
            if dN2 > 0: #check N-
                max_d2_to_stem = ((cenpos-half)*3.8)**2 + 9.0 # 3 Ang tolerance
                #print("threadN cen/n/d/cut: %3d %3d %5.2f %5.2f"%(cenpos,cenpos-half,
                #                                                  np.sqrt(dN2),np.sqrt(max_d2_to_stem)) )
                if dN2 > max_d2_to_stem: continue #skip this one
            
            if dC2 > 0: #check C-
                max_d2_to_stem = ((SS.nres-cenpos-1)*3.8)**2 + 9.0 # 3 Ang tolerance
                #print("threadC cen/n/d/cut: %3d %3d %5.2f %5.2f"%(cenpos,cenpos-half,
                #                                                  np.sqrt(dC2),np.sqrt(max_d2_to_stem)) )
                if dC2 > max_d2_to_stem: continue #skip this one

            # Then filter by rmsd-to-init -- look at N+1 to C-1
            pose_crd   = poseinfo.CAcrds[SS.cenres-half+1:SS.cenres+half]
            thread_crd = SS.CAcrds_al[cenpos-half+1:cenpos+half]
            dcrd = pose_crd-thread_crd
            rmsd_from_init = np.sqrt(np.sum(dcrd*dcrd)/len(SS.frame))

            #print("thread %d: rmsd %.3f (%.3f)"%(cenpos, rmsd_from_init, self.rmsdcut_from_init))
            if rmsd_from_init > self.rmsdcut_from_init:
                continue
            
            threads.append((cenpos-half,cenpos+half,cenpos)) #begin,end,cen
            
        return threads

    # UNUSED
    # a generic & compact output just storing resrange & bbcrds
    def report_as_npz(self,outf):
        threads = []
        bbcrds = []
        for (match,begin,end) in self.selected:
            threads.append([begin,end])
            bbcrds.append(match.bbcrd)
        np.savez(outf, threads=threads, bbcrds=bbrds)

    # Unused
    # replace & simplifies prv Rosetta app 
    def filter_by_score(self, poseinfo, ulrs, nmax ):
        crds_noulr = {}
        ulrres = []
        for ulr in ulrs: ulrres += ulr
        for res in poseinfo.reslist:
            if res not in ulrres: crds_noulr[res] = poseinfo.CBcrds[res]
        
        threadble = []
        for match in self.solutions: 
            threads = match.threads
            threadable += match.find_threads(estogram,SSpred)
        threadable.sort() #TODO
                
        covered = {}
        self.selected = []
        while npick < nmax:
            (score,match,begin,end) = threadable[0]
            self.selected.append((match,begin,end))
            covered[begin] += 1

            # add penalty to covered/picked
            for j,thread in enumerate(threadable):
                begin = thread[2]
                if j == 0:  thredable[j][0] += 9999 #score
                else: threadable[j][0] += covered[thread]*5.0
            threadable.sort() #resort reflecting penalty
