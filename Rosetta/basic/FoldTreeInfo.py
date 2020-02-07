import numpy as np
import rosetta_utils
import pyrosetta as PR
from PoseInfo import SSclass

class Jump:
    def __init__(self,anchor,cen,reslist,movable=True,is_ULR=False):
        self.anchor = anchor
        self.cen = cen
        self.reslist = reslist
        self.movable = movable
        self.subid = -1 #necessary?
        self.chunkid = -1 #necessary?
        self.is_ULR = is_ULR
        self.iSS = [] #list of SS index

        # default movement magnitude
        self.trans = 1.0
        self.rot = 5.0

    def estimate_RG_magnitude(self,maxdev):
        if not self.movable:
            self.trans = 0.0
            self.rot = 0.0

        else:
            self.trans = maxdev[self.cen]
            # simple logic for now
            maxtrans = np.max(maxdev[self.reslist[0]:self.reslist[-1]+1])

            if maxtrans < 2.0:
                self.maxrot = 5.0
            elif maxtrans < 4.0:
                self.maxrot = 15.0
            else:
                self.maxrot = 30.0
        
class FoldTreeInfo:
    def __init__(self,pose,opt):
        # static
        self.ft = pose.conformation().fold_tree().clone()
        if hasattr(opt,"min_chunk_len"):
            self.min_chunk_len = opt.min_chunk_len
        else:
            self.min_chunk_len = [3,5,1000] #E,H,L
            
        # default
        self.CAdev = [] #undefined
        self.subs = opt.subs
        self.ulrs = []

        # index in SSs_in that are not SS in the pose but should be built as SS
        self.jumps = []
        self.cuts = []
        self.movable = []
        self.jump2subid = [] #(subid,chunkid,resrange) for each jump

    def setup_fold_tree(self,pose,opt,ulrs=[],SSs_in=None,SS3types_in=None):
        self.ulrs = ulrs
        if opt.debug:
            print("\n[FoldTreeInfo.setup_fold_tree] ===== Setup FoldTree =====")
            
        if SSs_in == None or SS3types_in == None:
            ulrres = []
            for ulr in ulrs: ulrres += ulr
            # stores resrange and SS3types
            if opt.debug: print("\n** Define SS3 state from pose")
            self.SSs,self.SStypes = rosetta_utils.pose2SSs(pose,
                                                           maskres=ulrres, #forget info there
                                                           min_chunk_len=self.min_chunk_len,
                                                           report=opt.debug)
        else:
            self.SSs = SSs_in
            self.SStypes = SStypes_in
            
        if opt.SSpred != None:
            if opt.debug: print("\n** Get augmented-SS assignments from %s"%opt.SSpred)
            # TODO: move non-Rosetta related things to myutils...
            SS3pred = rosetta_utils.SS9p_to_SS3type(np.load(opt.SSpred)['ss9'])
            
            extraSSs,extraSStypes = rosetta_utils.extraSS_from_prediction(ulrs,SS3pred)

        extraSS_index = []
        if len(extraSSs) != []:
            extraSS_index = [len(self.SSs)+k for k in range(len(extraSSs))]
            self.SSs += extraSSs #still regions
            self.SStypes += extraSStypes
            # also store as SSclasses
            self.extraSS_at_ulr = []
            for i,reg in enumerate(extraSSs):
                self.extraSS_at_ulr.append( SSclass(extraSStypes[i],reg[0],reg[-1]) ) #no poseinfo

        # self.SSs & self.SStypes used
        self.jumps,self.cuts = self.setup_fold_tree_from_SS(pose,
                                                            extraSS_index=extraSS_index,
                                                            report=opt.debug)
        
    # SS is "augmented-SS" (from structure+pred)
    # ULR also should be defined as "SS" if pred says SS
    def setup_fold_tree_from_SS(self,pose,
                                extraSS_index=[],
                                report=False):

        ft = pose.conformation().fold_tree().clone()
        nres = pose.size()
        if pose.residue(nres).is_virtual_residue():
            nres -= 1

        ### should I rewrite as OOP? (so as to remove master_chunk_anchor sort of arrays...)
        #w/ Graph, Chunk class... (chunk.cen)
        
        # 0. get central position of SSs
        SScen = [-1 for k in self.SSs]
        for iSS,SS in enumerate(self.SSs):
            if iSS in extraSS_index:
                SScen[iSS] = SS[int(len(SS)/2)]
            else:
                SScen[iSS] = rosetta_utils.get_COMres(pose,SS)

        # 1. First get which SS belongs to which GraphChunk
        SS_at_subdef = {} # index of SS->graphchunk
        resrange_by_SS = [{} for sub in self.subs] #resrange defined by "clean" SSs whose center belongs to graphchunk
        master_chunk_anchor = {}
        if report: print("\n** Re-adjust sub/chunk definitions according to SS3 assignment")
            
        for i,sub in enumerate(self.subs):
            for j,chunk in enumerate(sub): #chunk cat be more than one SS
                resrange_by_SS[i][j] = []
                for iSS,SS in enumerate(self.SSs):
                    if SScen[iSS] in chunk:
                        SS_at_subdef[iSS] = (i,j)
                        resrange_by_SS[i][j] += SS
                if report: print("Re-adjusted region for %d.%d: %d-%d -> %d-%d"%(i,j,chunk[0],chunk[-1],resrange_by_SS[i][j][0],resrange_by_SS[i][j][-1]))
                
                if j == 0: #master chunk
                    cen = rosetta_utils.get_COMres(pose,resrange_by_SS[i][j])
                    master_chunk_anchor[i] = cen
            
        # 2. Define jumps: scan through SS
        if report: print("\n** Jump/Cut definitions")
        subchunk_covered = [] #skip duplication (e.g. multi SS def. for 1 sub-chunk)
        jumps = []
        for iSS,SS in enumerate(self.SSs):
            print("??", iSS, SS)
            if iSS in SS_at_subdef: # SS that belongs to Graph
                (i,j) = SS_at_subdef[iSS]
                # once per multi-SS-jump
                if (i,j) in subchunk_covered: continue 

                resrange = resrange_by_SS[i][j]
                #print("i.j/resrange %d.%d/%d-%d"%(i,j,resrange[0],resrange[-1]))
                cen = rosetta_utils.get_COMres(pose,resrange)
                
                if j == 0: anc = nres+1 #MasterChunk
                else: anc = master_chunk_anchor[i] #BranchChunk
                
                jump = Jump(anc,cen,resrange,movable=False,is_ULR=False)
                # append all SS index sharing same chunk index
                for iSS in SS_at_subdef:
                    if SS_at_subdef[iSS] == (i,j):
                        jump.iSS.append(iSS)
                subchunk_covered.append((i,j))
                
            else: # rest
                resrange = SS
                cen = rosetta_utils.get_COMres(pose,SS)
                anc = nres+1 #Vroot

                jump = Jump(anc,cen,resrange,movable=True,
                            is_ULR=(iSS in extraSS_index))
                jump.iSS = [iSS]
            print("Jump %d (%3d-%3d), %3d-> %3d"%(len(jumps),
                                                  resrange[0],resrange[-1],cen,anc))
            jumps.append(jump)

        # 3. Find cut points b/w jumps
        jumpres = [jump.reslist for jump in jumps]
        jumpres.sort()
        cuts = []
        for i,reslist in enumerate(jumpres[:-1]):
            cut = jumpres[i+1][0] #beginning of next jump res
            cuts.append(cut) 
        cuts.append(nres) 
        
        # Rosetta jump definition: anchorres & cenres
        jumpdef = [(jump.anchor,jump.cen) for jump in jumps]
        print( "Cuts: ", cuts)

        stat = rosetta_utils.tree_from_jumps_and_cuts(ft,nres+1,jumpdef,cuts,nres+1)
        
        if not pose.residue(pose.size()).is_virtual_residue(): 
            PR.rosetta.core.pose.addVirtualResAsRoot(pose)

        PR.rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        PR.rosetta.protocols.loops.add_cutpoint_variants( pose )
        return jumps, cuts
        
    ## manual
    def setup_fold_tree_from_defined_subs(self,pose,subs):
        ft = pose.conformation().fold_tree().clone()
        nres = pose.size()
        if pose.residue(nres).is_virtual_residue():
            nres -= 1

        jumps = []
        cuts = []
        for sub in subs:
            for i,chunk in enumerate(sub):
                chunkcen = get_COMres(pose,chunk)
                cuts.append(chunk[-1])
                if i == 0:
                    ancres = chunkcen
                    jumps.append(Jump(nres+1,chunkcen,chunk,False)) #place jump at central res to vroot
                else:
                    jumps.append(Jump(ancres,chunkcen,chunk,True)) #place jump at central res to sub-anchor
                
        cuts.append(nres)
    
        stat = tree_from_jumps_and_cuts(ft,nres+1,jumps,cuts,nres+1)
        
        if nres == pose.size():
            rosetta.core.pose.addVirtualResAsRoot(pose)
    
        rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        rosetta.protocols.loops.add_cutpoint_variants( pose )
        return jumps

    '''
    def extraSS_at_ulr(self,poseinfo):
        jumps_return = []
        for jump in self.jumps:
            if jump.is_ULR:
                reslist = 
                jumps_return.append( jump ) #extraSS should only have 1 SS-index
        return jumps_return #list of range
    '''

    def estimate_RG_magnitudes(self,maxdev):
        for jump in self.jumps:
            jump.estimate_RG_magnitude()

    def get_jumpid(self, reslist):
        jumpid = -1
        for i,jump in enumerate(self.jumps):
            if jump.cen in reslist:
                if jumpid != -1:
                    return -1 # more than one jump possible -- return "false"
                jumpid = i
        return jumpid

    
