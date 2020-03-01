import numpy as np
import rosetta_utils
import pyrosetta as PR
from PoseInfo import SSclass
import random
import estogram2FT

# Data storage defining movements of the jump it belongs to
class JumpMove:
    def __init__(self,tag=''):
        self.tag = tag
        self.tors = {} #key as pose resno
        self.stub_SS = []
        self.stub_SS_np = []
    
class Jump:
    def __init__(self,anchor,cen,reslist,movable=True,is_ULR=False):
        self.anchor = anchor
        self.cen = cen
        self.movable = movable
        self.reslist = list(reslist)
        self.is_ULR = is_ULR
        
        # default movement magnitude
        self.trans = 1.0
        self.rot = 5.0

        # moveset related
        self.movesets = []
        self.stub_anc = None
        self.stub_anc_np = []

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

    def append_moveset_from_match(self,pose,solution,
                                  threadable_cenpos,
                                  iSS=-1): #index of SS altered
        
        # Anchor info shared across movesets
        if self.stub_anc_np == []:
            vroot = pose.residue(pose.size())
            self.stub_anc = PR.rosetta.core.kinematics.Stub(vroot.xyz("ORIG"),vroot.xyz("X"),vroot.xyz("Y"))
            self.stub_anc_np = np.array([[vroot.xyz("ORIG")[0],vroot.xyz("ORIG")[1],vroot.xyz("ORIG")[2]],
                                         [vroot.xyz("X")[0]   ,vroot.xyz("X")[1]   ,vroot.xyz("X")[2]],
                                         [vroot.xyz("Y")[0]   ,vroot.xyz("Y")[1]   ,vroot.xyz("Y")[2]]])

        solutionSS = solution.SSs[iSS] #support only 1 SS for now...
        
        xyz_n  = pose.residue(1).xyz("N") #placeholder
        xyz_ca = pose.residue(1).xyz("CA")#placeholder 
        xyz_cp = pose.residue(1).xyz("C") #placeholder
        
        # Define jump from stubs: solutionSS.cenres  == self.cen??
        for cenpos in threadable_cenpos:
            shift = cenpos - solutionSS.cenpos
            
            moveset = JumpMove(tag=solution.tag+".t%02d"%(cenpos))
            for k in range(3):
                xyz_n[k]  = solutionSS.bbcrds_al[cenpos][0][k]
                xyz_ca[k] = solutionSS.bbcrds_al[cenpos][1][k]
                xyz_cp[k] = solutionSS.bbcrds_al[cenpos-1][2][k]
                
            moveset.stub_SS    = PR.rosetta.core.kinematics.Stub(xyz_ca,xyz_n,xyz_cp)
            moveset.stub_SS_np = np.array([solutionSS.bbcrds_al[cenpos][0],
                                           solutionSS.bbcrds_al[cenpos][1],
                                           solutionSS.bbcrds_al[cenpos-1][2]])

            moveset.tors = {}
            for ires in range(solutionSS.nres):
                resno = solutionSS.begin + ires + shift
                if resno > 0 and resno < pose.size():
                    moveset.tors[resno] = solutionSS.tors[ires]
                
            self.movesets.append(moveset)

    def moveset_to_pose(self,pose_in,mode='random',report_pdb=False):
        if mode == 'random':
            movesets = [random.choice(self.movesets)]
        elif mode == 'all':
            movesets = self.movesets

        poses = []
        for moveset in movesets:
            pose = pose_in #copy

            newjump = PR.rosetta.core.kinematics.Jump(self.stub_anc,moveset.stub_SS)
            pose.conformation().set_jump(self.ijump+1,newjump) #ijump is 0-index

            for resno in moveset.tors:
                phi,psi,omg = moveset.tors[resno]
                if abs(phi) < 360.0: pose.set_phi(resno,phi)
                if abs(psi) < 360.0: pose.set_psi(resno,psi)
                if abs(omg) < 360.0: pose.set_omega(resno,omg)
            poses.append(pose)
            if report_pdb: pose.dump_pdb(moveset.tag+".pdb")
        return poses

    def report_as_npz(self,outf):
        return


class FoldTreeInfo:
    def __init__(self,pose,opt): 
        # static
        self.ft = pose.conformation().fold_tree().clone()
        if hasattr(opt,"min_chunk_len"):
            self.min_chunk_len = opt.min_chunk_len
        else:
            self.min_chunk_len = [3,5,1000] #E,H,L
        self.nres = pose.size()
        if pose.residue(self.nres).is_virtual_residue(): self.nres -= 1

        self.subs = []

        ## Info being stored/retrieved
           
        # i) directly from estogram
        self.Qres =  [] #predicted l-DDT
        self.Qres_corr = [] #corrected Q-res
        self.maxdev = []
        # ii) processed given options
        self.jumps = []
        self.cuts = []
        self.SSs_reg = []
        self.SSs_ulr = []

    def init_from_estogram(self,pose,opt,
                           poseinfo=None):
        data = np.load(opt.npz)
        estogram  = data['estogram']
        self.Qres = data['lddt']
        self.Q    = np.mean(self.Qres)
        if opt.subdef_confidence_offset == "auto":
            opt.subdef_threshould = self.Q + 0.1
        elif isinstance(opt.subdef_confidence_offset,float):
            opt.subdef_threshould = self.Q + opt.subdef_confidence_offset

        estogram2FT.add_default_options_if_missing(opt)

        if poseinfo == None: poseinfo = PoseInfo(pose,opt,SSpred_fn=opt.SSpred_fn)

        # 1. Predict ULR
        if opt.debug: print("\n[estogram2FT] ========== ULR prediction ========")
        self.ulrs, self.Qres_corr = estogram2FT.estogram2ulr(estogram,opt)
        self.ulrres = []
        for ulr in self.ulrs: self.ulrres += ulr
    
        # 1-a. assign maximum allowed deviation in coordinates
        self.maxdev = np.zeros(len(estogram))
        for i,val in enumerate(self.Qres_corr):
            self.maxdev[i] = estogram2FT.Qres2dev(val) #fitting function
        #self.estimate_RG_magnitudes(maxdev) # later
        
        if opt.debug:
            print( "ULR detected,mode=%s: "%opt.ulrmode)
            for ulr in self.ulrs:
                ulrconf = np.mean(self.Qres[ulr[0]-1:ulr[-1]]) #confidence of current struct
                print( "%d-%d: confidence %.3f"%( ulr[0],ulr[-1],ulrconf) )

        # 2. Predict subs -- multiple variable subs
        if opt.debug: print("\n[estogram2FT] ========== SubChunk assignment ========")

        self.subs = estogram2FT.estogram2sub(estogram, poseinfo.SS3_naive, self.ulrres, opt)

        # 3. Finally setup fold tree given sub & ulr info
        self.setup_fold_tree(pose,opt,poseinfo)
    
    def setup_fold_tree(self,pose,opt,poseinfo):
        if opt.debug:
            print("\n[FoldTreeInfo.setup_fold_tree] ===== Setup FoldTree =====")

        ## Assign ULR-SSs
        # i) first check if any assigned SS belongs to ULR-SS
        self.SSs_reg = []
        self.SSs_ulr = []
        for SS in poseinfo.SSs:
            if SS.cenres in self.ulrres: self.SSs_ulr.append(SS)
            else: self.SSs_reg.append(SS)

        # ii) then check missing ones from pred
        if opt.SSpred_fn != None:
            SS3pred = rosetta_utils.SS9p_to_SS3type(np.load(opt.SSpred_fn)['ss9'])
            if opt.debug:
                print("\n** Get augmented-SS assignments from %s"%opt.SSpred_fn)
                print("SS3fromPred: ", ''.join(SS3pred))

            # TODO: move non-Rosetta related things to myutils...
            extraSSs,extraSStypes = rosetta_utils.extraSS_from_prediction(self.ulrs,SS3pred)

        if len(extraSSs) != []:
            for i,reg in enumerate(extraSSs):
                ulrSS = SSclass(extraSStypes[i],reg[0],reg[-1])
                ulrSS.is_ULR = True
                self.SSs_ulr.append( ulrSS ) #no poseinfo

        # iii) define self.jumps & self.cuts that can be translated to Rosetta jump def
        self.setup_fold_tree_from_SS(report=opt.debug)

        # iv) Finally rosetta stuffs once defined self.cuts, self.jumps
        ft = pose.conformation().fold_tree().clone()
        jumpdef = [(jump.anchor,jump.cen) for jump in self.jumps]
        print(len(self.cuts))
        print(len(self.jumps))
        stat = rosetta_utils.tree_from_jumps_and_cuts(ft,self.nres+1,jumpdef,self.cuts,self.nres+1)

        if not pose.residue(pose.size()).is_virtual_residue(): 
            PR.rosetta.core.pose.addVirtualResAsRoot(pose)

        PR.rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        PR.rosetta.protocols.loops.add_cutpoint_variants( pose )
        
    ### Use self.SSs_reg & self.SS_ulr below
    def setup_fold_tree_from_SS(self,report=False):
        SSs_all = self.SSs_reg + self.SSs_ulr # sort by reg/ulr SSs
        n_nonulrSS = len(self.SSs_reg)
        
        if report: print("\n** Re-adjust sub/chunk definitions according to SS3 assignment")

        ## 1. First get which SS belongs to which GraphChunk (grab info from self.subs)
        iSSs_at_sub = [[] for sub in self.subs]
        for iSS,SS in enumerate(SSs_all[:n_nonulrSS]):
            cenres = SSs_all[iSS].cenres
            for i,sub in enumerate(self.subs):
                iSSs_at_sub[i] += [j for j,chunk in enumerate(sub) if cenres in chunk]

        # Re-define master having the longest length -- forget "j==0" (original designation by estogram2sub)
        sub_redef = [-1 for SS in SSs_all] # -1 as master
        for isub,iSSs in enumerate(iSSs_at_sub):
            if len(iSSs) <= 1: continue # treat as regular
            imaster = np.argmax([SSs_all[iSS].nres for iSS in iSSs])
            for iSS in iSSs:
                if iSS != imaster: sub_redef[iSS] = imaster #rest as branch
                    
        ## 2. Define jumps: scan through SS
        self.jumps = []
        for iSS,SS in enumerate(SSs_all):
            j = sub_redef[iSS] #master iSS index
            if j == -1: anc = self.nres+1             #MasterChunk
            else:       anc = SSs_all[j].cenres #BranchChunk
            movable = (anc != self.nres+1)
            jump = Jump(anc,SS.cenres,SS.reslist,movable=movable,is_ULR=(iSS >= n_nonulrSS))
                
            print("Jump %d (%3d-%3d), %3d-> %3d"%(len(self.jumps),SS.begin,SS.end,
                                                  SS.cenres,anc))
            jump.ijump = len(self.jumps) #necessary?
            self.jumps.append(jump)

        ## 3. Find cut points b/w jumps 
        jumpres = [jump.reslist for jump in self.jumps]
        print("jumpres?", jumpres)
        jumpres.sort()
        self.cuts = []
        for i,reslist in enumerate(jumpres[:-1]):
            cut = jumpres[i+1][0] #beginning of next jump res
            self.cuts.append(cut) 
        self.cuts.append(self.nres) 
        print( "Cuts: ", self.cuts)
        
    ## UNUSED: manual
    def setup_fold_tree_from_defined_subs(self,pose,subs):
        ft = pose.conformation().fold_tree().clone()
        self.nres = pose.size()
        if pose.residue(self.nres).is_virtual_residue():
            self.nres -= 1

        jumps = []
        cuts = []
        for sub in subs:
            for i,chunk in enumerate(sub):
                chunkcen = get_COMres(pose,chunk)
                cuts.append(chunk[-1])
                if i == 0:
                    ancres = chunkcen
                    jumps.append(Jump(self.nres+1,chunkcen,chunk,False)) #place jump at central res to vroot
                else:
                    jumps.append(Jump(ancres,chunkcen,chunk,True)) #place jump at central res to sub-anchor
                
        cuts.append(self.nres)
    
        stat = tree_from_jumps_and_cuts(ft,self.nres+1,jumps,cuts,self.nres+1)
        
        if self.nres == pose.size():
            rosetta.core.pose.addVirtualResAsRoot(pose)
    
        rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        rosetta.protocols.loops.add_cutpoint_variants( pose )
        return jumps

    def save_as_npz(self,outf):
        regjumpinfo = np.ndarray((len(self.SSs_reg),6)) #begin,end,cut,cen,anc,is_ulr
        # 1. residue indices defining jump & cut
        for i,SS in enumerate(self.SSs_reg):
            jump = self.jumps[i]
            jumpinfo[i] = np.ndarray([SS.begin,SS.end,self.cuts[i],jump.cen,jump.anchor,0])
            
        ulrjumpinfo = np.ndarray((len(self.SSs_ulr),6)) #begin,end,cut,cen,anc,is_ulr

        # candidates of ULR & vroot stubs
        ulrstub = [{} for ulr in self.SSs_ulr]
        for i,SS in enumerate(self.SSs_ulr):
            ij = i+len(self.SSs_reg)
            jump = self.jumps[ij]
            ulrinfo[ij] = np.ndarray([SS.begin,SS.end,self.cuts[ij],jump.cen,jump.anchor,1])
            
            ulrstub[i]['anchor'] = jump.stub_anc_np
            ulrstub[i]['ulr']    = [moveset.stub_SS_np for moveset in jump.movesets]
            
        np.savez(outf,
                 Qres=self.Qres,
                 Qres_corr=self.Qres_corr,
                 maxdev=self.maxdev,
                 regjump=regjumpinfo,
                 ulrjump=ulrjumpinfo,
                 vrootstub=vrootstub,
                 ulrstubs=ulrstubs)

    def load_from_npz(self,npz,pose):
        if not pose.residue(pose.size()).is_virtual_residue(): 
            PR.rosetta.core.pose.addVirtualResAsRoot(pose)
        self.nres = pose.size()-1

        info = np.load(np)

        # Info to fill in
        self.Qres = info['Qres']
        self.Qres_corr = info['Qres_corr']
        self.SSs_ulr = []
        self.jumps = []
        self.cuts = []
        
        if self.nres != len(self.Qres):
            sys.exit("Failed to retrieve FTinfo from file %s: nres inconsistent!"%npz)
        
        # first, load jumps and cuts
        self.cuts = []
        self.jumps = []
        for [b,e,cut,cen,anc,is_ulr] in info['regjump']:
            movable = (anc != nres+1)
            jump = Jump(anc,cen,range(b,e+1),movable=movable,is_ULR=0)
            self.jumps.append(jump)
            self.cuts.append(cut)
        for [b,e,cut,cen,anc,is_ulr] in info['ulrjump']:
            jump = Jump(anc,cen,range(b,e+1),movable=1,is_ULR=1)
            self.jumps.append(jump)
            self.cuts.append(cut)
        
        ft = pose.conformation().fold_tree().clone()
        jumpdef = [(jump.anchor,jump.cen) for jump in self.jumps]
        stat = rosetta_utils.tree_from_jumps_and_cuts(ft,nres,jumpdef,
                                                      self.nres+1,self.cuts,self.nres+1)

        PR.rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        PR.rosetta.protocols.loops.add_cutpoint_variants( pose )

        # (optional) load stub set if exists
        # load placeholders first
        xyz1  = pose.residue(pose.size()).xyz("ORIG")
        xyz2  = pose.residue(pose.size()).xyz("X")
        xyz3  = pose.residue(pose.size()).xyz("Y")
        for i,jump in enumerate(self.jumps):
            jump.stub_anc_np = info['ulrstubs'][i]['anchor']
            for k in range(3):
                xyz1[k] = jump.stub_anc_np[0,k]
                xyz2[k] = jump.stub_anc_np[1,k]
                xyz3[k] = jump.stub_anc_np[2,k]
            jump.stub_SS = PR.rosetta.core.kinematics.Stub(xyz1,xyz2,xyz3)

            for stub_np in info['ulrstub'][i]['ulr']:
                moveset = JumpMove()
                moveset.stub_SS_np = info['ulrstubs'][i]['ulr']
                for k in range(3):
                    xyz1[k] = stub_np[0,k]
                    xyz2[k] = stub_np[1,k]
                    xyz3[k] = stub_np[2,k]
                moveset.stub_SS = PR.rosetta.core.kinematics.Stub(xyz1,xyz2,xyz3)
                jump.movesets.append(moveset)
    
    def estimate_RG_magnitudes(self,maxdev):
        for jump in self.jumps:
            jump.estimate_RG_magnitude()

    def get_jumpid(self, reslist):
        jumpid = -1
        for i,jump in enumerate(self.jumps):
            if jump.cen in reslist:
                if jumpid != -1:
                    return -1 # more than one jump possible
                jumpid = i
        return jumpid
    
    def get_jump(self,reslist):
        jumpid = self.get_jumpid(reslist)
        if jumpid == -1: return False
        return self.jumps[jumpid]
        
