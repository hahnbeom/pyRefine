import numpy as np

## "jump": Term or MatchSolution class
def distance(crd1,crd2):
    dv = crd1-crd2
    return np.sqrt(np.sum(dv*dv))

# Inherits prv. "JumpDBtype"
class Matcher:
    def __init__(self,db,opt,prefix,debug=False):
        self.prefix = prefix
        self.db = db #TermDB, for instance
        self.report_pdb = debug #should move to opt
        self.duplication_rmsd_cut = opt.config['DUPLICATION_RMSD'] #5.0
        self.solutions = []
        self.rmsd_anchor_cut = opt.config['RMSD_ANCHOR_CUT'] #2.0
        self.clash_cut = opt.config['MAX_CLASH_COUNT']

    # previously called "scan_through_generic"
    def place_ULR_at_anchors(self,poseinfo,anchors,ulr_t, #ulr_t should be defined as SSclass
                             include_hydrophobic=0): 
        print( "\nULR %d-%d, scan through %d terms:"%(ulr_t.begin,ulr_t.end,len(anchors)) )

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
            d_jump_to_ulranc = distance(jumpcom,anccrd)
            nres_linker = len(ulrres)-3 # rough estimation of max length
            if d_jump_to_ulranc > dperres*nres_linker:
                return False
            else:
                return True

    ## originally called scan_regions...
    def find_matches_through_DB(self,poseinfo,anchor,ulr_t,
                                outprefix='',report_as_a_file=True):
        seqpos = ulr_t.reslist[0] # starting seqpos

        #solutions = self.db.search_similar(anchor,ulr_t)
        query = anchor + [ulr_t]
        solutions = []
        for term in self.db.db:
            solutions += term.find_solutions(query,self.rmsd_anchor_cut)

        if solutions == []: return solutions
            
        print("GOT %d solutions"%len(solutions))

        # filter
        filtered = []
        for i,match in enumerate(solutions):
            match.transrot_bbcrd()
            if poseinfo.SS_is_clashing(match.SSs_term[-1],clash_cut=self.clash_cut):
                match.write_as_pdb("clash.%s.pdb"%match.tag)
            else:
                match.write_as_pdb("legit.%s.pdb"%match.tag)
                filtered.append(match)
        return filtered
    
        ##TODO
        for isol,jump_al in enumerate(solutions):
            # filter by clash
            if poseinfo.SS_is_clashing(jump_al.SSs[-1],clash_cut=CLASHCUT):
                nclash += 1
                continue

            # thread search -- map ULR-SS cenres to TERM cenres
            #threads,bestmatch = ulr.thread_match(jump_al.SSs[-1],seqscorecut=SEQSCORECUT)
            #jump_and_thread_solutions.append((jump_al,threads,isol))

        # Select "threads" -- simplify to just keep above??
        jump_and_thread_solutions = []
        ndupl = 0
        nmatches = 0
        selectedSS = []
        self.chunkmatches = []
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
                
                if self.report_pdb: pdbname = 'match.%s.%s.%d.pdb'%(outprefix,jump_al.index,isol)
                else: pdbname = ''

                outfile=None
                if report_as_a_file:
                    outfile = self.out
                    
                self.chunkmatches += jump_al.report_chunk_insertion(RefJump,RefJump.nSS-1, #always sorted as final
                                                              threads, isol, out=outfile,
                                                              report_as_pdb=pdbname
                )
                selectedSS.append(SSreplaced)

        print( "--  Match %s: Total %3d jump solutions found (%d/%d filtered from %d by clash/struct-similarity"%(outprefix,len(selected),nclash,ndupl,nmatches) )

    # replace & simplifies prv Rosetta app 
    def do_filter(self, poseinfo, ulrs, nmax ):
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

    def lookup(self,vals,d,dbins):
        ibin = min(max(0,int(d)-3),len(vals)-1)
        for dbin in dbins:
            if ibin < dbin: return vals[dbin]
        return 0.0

    # a generic & compact output just storing resrange & bbcrds
    def report_as_npz(self,outf):
        threads = []
        bbcrds = []
        for (match,begin,end) in self.selected:
            threads.append([begin,end])
            bbcrds.append(match.bbcrd)
        np.savez(outf, threads=threads, bbcrds=bbrds)

