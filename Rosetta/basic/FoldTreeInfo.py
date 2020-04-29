import sys
import numpy as np
import rosetta_utils
import pyrosetta as PR
from PoseInfo import SSclass
import random
import estogram_utils

# Data storage defining movements of the jump it belongs to
class JumpMove:
    def __init__(self,tag=''):
        self.tag = tag
        self.tors = {} #key as pose resno
        self.stub_SS = []
        self.stub_SS_np = []
    
class Jump:
    def __init__(self,anchor,cen,reslist,movable=True,is_ULR=False,ijump=-1,Q=0.0):
        self.anchor = anchor
        self.cen = cen
        self.movable = movable
        self.reslist = list(reslist)
        self.nres = len(reslist)
        self.is_ULR = is_ULR
        self.Q = Q
        
        # default movement magnitude -- unused
        self.trans = 1.0
        self.rot = 5.0
        self.ijump = ijump

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
        
        # Define jump from stubs: solutionSS.cenres  == self.cen??
        for cenpos in threadable_cenpos:
            moveset = JumpMove(tag=solution.tag+".t%02d"%(cenpos))

            xyz_n  = PR.rosetta.numeric.xyzVector_double_t(0.0)
            xyz_ca = PR.rosetta.numeric.xyzVector_double_t(0.0)
            xyz_cp = PR.rosetta.numeric.xyzVector_double_t(0.0)
            for k in range(3):
                xyz_ca[k] = solutionSS.bbcrds_al[cenpos][1][k]
                xyz_n[k]  = solutionSS.bbcrds_al[cenpos][0][k]
                xyz_cp[k] = solutionSS.bbcrds_al[cenpos-1][2][k]
                
            moveset.stub_SS    = PR.rosetta.core.kinematics.Stub(xyz_ca,xyz_n,xyz_cp)
            moveset.stub_SS_np = np.array([solutionSS.bbcrds_al[cenpos][1], #CA
                                           solutionSS.bbcrds_al[cenpos][0], #N 
                                           solutionSS.bbcrds_al[cenpos-1][2]]) #Cprv

            moveset.tors = {}
            shift = cenpos - solutionSS.cenpos
            for ires in range(solutionSS.nres):
                resno = solutionSS.begin + ires + shift
                if resno > 0 and resno < pose.size():
                    moveset.tors[resno] = solutionSS.tors[ires]
                
            self.movesets.append(moveset)

    def moveset_to_pose(self,pose_in,mode='random',report_pdb=False,nmax=100):
        if mode == 'random':
            movesets = [random.choice(self.movesets)]
        elif mode == 'all':
            movesets = self.movesets

        poses = []
        restags = []
        for moveset in movesets:
            pose = pose_in.clone() #copy
            # make sure C-term is vroot
            vroot = pose.residue(pose.size())
            if not vroot.is_virtual_residue():
                print("Vroot not found!! continue")
            stub_anc = PR.rosetta.core.kinematics.Stub(vroot.xyz("ORIG"),vroot.xyz("X"),vroot.xyz("Y"))
            
            newjump = PR.rosetta.core.kinematics.Jump(stub_anc,moveset.stub_SS)
            pose.conformation().set_jump(self.ijump+1,newjump) #ijump is 0-index

            # write tors from TERM
            for resno in moveset.tors:
                phi,psi,omg = moveset.tors[resno]
                if abs(phi) < 360.0: pose.set_phi(resno,phi)
                if abs(psi) < 360.0: pose.set_psi(resno,psi)
                if abs(omg) < 360.0: pose.set_omega(resno,omg)
            poses.append(pose)
            restags.append(moveset.tors.keys())
            if report_pdb: pose.dump_pdb(moveset.tag+".pdb")
            if len(poses) >= nmax: break
            
        return poses,restags

class FoldTreeInfo:
    def __init__(self,pose,opt): 
        # static
        if hasattr(opt,"min_chunk_len"):
            self.min_chunk_len = opt.min_chunk_len
        else:
            self.min_chunk_len = [3,5,1000] #E,H,L

        self.nres = 0
        if pose != None:
            self.ft = pose.conformation().fold_tree().clone()
            self.nres = pose.size()
            if pose.residue(self.nres).is_virtual_residue(): self.nres -= 1

        self.allow_beta_jump = False
        if hasattr(opt,"allow_beta_jump"):
            self.allow_beta_jump = opt.allow_beta_jump

        self.subdefs = []
        self.cuttype = 'end'
        if hasattr(opt,"cut_at_mid"):
            if getattr(opt,"cut_at_mid"): self.cuttype = 'mid'

        ## Info being stored/retrieved
           
        # i) directly from estogram
        self.Qres =  [] #predicted l-DDT
        self.Qres_corr = [] #corrected Q-res
        self.Qtors = [] # torsion confidence
        self.maxdev = []
        # ii) processed given options
        self.jumps = []
        self.cuts = []
        self.SSs_reg = []
        self.SSs_ulr = []
        self.ulrs = []
        self.ulrs_aggr = []
        self.ulrs_cons = []
        self.stub_anc_np = None

    # this function defines only ulrs
    def init_from_estogram(self,pose,opt,
                           npz=None,
                           poseinfo=None,
                           subdef_confidence_offset="auto"):
        if npz == None: npz = opt.npz
        data = np.load(npz)
        estogram  = data['estogram']
        self.Qres = data['lddt']
        self.Q    = np.mean(self.Qres)
        
        estogram_utils.add_default_options_if_missing(opt)

        if poseinfo == None: poseinfo = PoseInfo(pose,opt,SSpred_fn=opt.SSpred_fn)

        # 1. Predict ULR
        # basic definition of ulrs
        if opt.debug: print("\n[FTInfo] ========== ULR prediction ========")
        self.ulrs_aggr, self.ulrs_cons, self.Qres_corr = estogram_utils.estogram2ulr(estogram,opt)
    
        # 1-a. assign maximum allowed deviation in coordinates
        self.maxdev = np.zeros(len(estogram))
        for i,val in enumerate(self.Qres_corr):
            self.maxdev[i] = estogram_utils.Qres2dev(val) #fitting function
        #self.estimate_RG_magnitudes(maxdev) # later
        
        if opt.debug:
            print( "ULR detected,mode=aggr: ")
            for ulr in self.ulrs_aggr:
                ulrconf = np.mean(self.Qres[ulr[0]-1:ulr[-1]]) #confidence of current struct
                print( "%3d-%3d: confidence %.3f"%( ulr[0],ulr[-1],ulrconf) )
            print( "\nULR detected,mode=cons: ")
            for ulr in self.ulrs_cons:
                ulrconf = np.mean(self.Qres[ulr[0]-1:ulr[-1]]) #confidence of current struct
                print( "%3d-%3d: confidence %.3f"%( ulr[0],ulr[-1],ulrconf) )

        # 2. Predict subs -- variable sub defs...
        if opt.do_subdef:
            if subdef_confidence_offset == "auto":
                subdef_base = self.Q + 0.1
            elif isinstance(subdef_confidence_offset,float):
                subdef_base = self.Q + subdef_confidence_offset

            if opt.debug: print("\n[FTInfo] ========== SubChunk assignment ========")
            ulrres = []
            for ulr in self.ulrs_aggr: ulrres += ulr
            
            # should have possible connections, not a strict definition
            for confcut in [subdef_base-0.2,subdef_base-0.1,subdef_base,subdef_base+0.1]:
                subs = estogram_utils.estogram2sub(estogram,
                                                   poseinfo.SS3_naive,
                                                   ulrres, #from aggr
                                                   opt,
                                                   confcut=confcut,
                                                   out=sys.stdout)
                #print(confcut,subs)
                # append only if at least one sub
                #print(subs)
                if len(subs) > 0:
                    self.subdefs.append(subs)
            
        # Setup fold tree given sub & ulr info -- move outside for variable defs
        #self.setup_fold_tree(pose,opt,poseinfo,ulrs=self.ulrs)
    
    def setup_fold_tree(self,pose,opt,poseinfo,
                        subdef_in=[],ulrs=None,
                        max_pert_jump_loc=0):
        if opt.debug:
            print("\n[FoldTreeInfo.setup_fold_tree] ===== Setup FoldTree =====")

        if ulrs == None: ulrs = self.ulrs_cons
        ulrres = []
        for ulr in ulrs: ulrres += ulr
        
        self.ulrs = ulrs #cache

        if max_pert_jump_loc > 0:
            for SS in poseinfo.SSs:
                # pert each SS randomly
                if random.random() > 0.5: continue
                
                while True:
                    shift = random.choice(range(-max_pert_jump_loc,max_pert_jump_loc+1))
                    if SS.cenres+shift in SS.reslist[1:-1]:
                        SS.cenres += shift
                        break
        
        ## Assign ULR-SSs
        # i) first check if any assigned SS belongs to ULR-SS
        self.SSs_reg = [] # cached for TERM?
        self.SSs_ulr = [] # cached for TERM?
        SSmask = []
        for SS in poseinfo.SSs:
            if SS.cenres in ulrres: self.SSs_ulr.append(SS)
            else: self.SSs_reg.append(SS)
            SSmask += SS.reslist

        # ii) then check missing ones from pred
        if opt.SSpred_fn != None:
            SS3pred = rosetta_utils.SS9p_to_SS3type(np.load(opt.SSpred_fn)['ss9'])
            if opt.debug:
                print("\n** Get augmented-SS assignments from %s"%opt.SSpred_fn)
                print("SS3fromPred: ", ''.join(SS3pred))

            # TODO: move non-Rosetta related things to myutils...
            extraSSs,extraSStypes = rosetta_utils.extraSS_from_prediction(self.ulrs,SS3pred,mask_in=SSmask)

        if len(extraSSs) != []:
            for i,reg in enumerate(extraSSs):
                ulrSS = SSclass(extraSStypes[i],reg[0],reg[-1])
                ulrSS.is_ULR = True
                self.SSs_ulr.append( ulrSS ) #no poseinfo

        # iii) define self.jumps & self.cuts that can be translated to Rosetta jump def
        cuts,jumps = self.setup_fold_tree_from_SS(subdef_in=subdef_in,
                                                  SSs_ulr=self.SSs_ulr,
                                                  report=opt.debug)
        self.cuts = cuts   # cache
        self.jumps = jumps # cache
        
        # iv) Finally rosetta stuffs once defined self.cuts, self.jumps
        ft = pose.conformation().fold_tree().clone()
        jumpdef = [(jump.anchor,jump.cen) for jump in jumps]
        stat = rosetta_utils.tree_from_jumps_and_cuts(ft,self.nres+1,jumpdef,cuts,self.nres+1)
        if not stat:
            print("Failed to generate FT from cut/jump def!")
            print(" - cuts: ", cuts)
            print(" - jumps (anchor/cen): ", jumpdef)
            sys.exit()

        if not pose.residue(pose.size()).is_virtual_residue(): 
            PR.rosetta.core.pose.addVirtualResAsRoot(pose)

        PR.rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        PR.rosetta.protocols.loops.add_cutpoint_variants( pose )

        # store vroot
        vroot = pose.residue(pose.size())
        self.stub_anc_np = np.array([[vroot.xyz("ORIG")[0],vroot.xyz("ORIG")[1],vroot.xyz("ORIG")[2]],
                                     [vroot.xyz("X")[0]   ,vroot.xyz("X")[1]   ,vroot.xyz("X")[2]],
                                     [vroot.xyz("Y")[0]   ,vroot.xyz("Y")[1]   ,vroot.xyz("Y")[2]]])
        
        
    ### Use self.SSs_reg & self.SS_ulr below
    def setup_fold_tree_from_SS(self,subdef_in=[],SSs_ulr=None,report=False):

        if SSs_ulr == []: SSs_ulr = self.SSs_ulr
        SSs_all = self.SSs_reg + SSs_ulr # sort by reg/ulr SSs
        n_nonulrSS = len(self.SSs_reg)

        if report: print("\n** Re-adjust sub/chunk definitions according to SS3 assignment")

        ## 1. First get which SS belongs to which GraphChunk 
        iSSs_at_sub = [[] for sub in subdef_in]
        nres_at_sub = [0 for sub in subdef_in]
        for iSS,SS in enumerate(SSs_all[:n_nonulrSS]):
            cenres = SSs_all[iSS].cenres
            for isub,sub in enumerate(subdef_in):
                SS_at_sub = []#chunk for chunk in sub if cenres in chunk]
                for chunk in sub:
                    if cenres in chunk:
                        iSSs_at_sub[isub].append(iSS)
                        nres_at_sub[isub] += SS.nres
                    #print("isub %d: SS_at_sub around %d"%(isub,cenres))

        master_sub = -1 #unused if remains as is
        if len(nres_at_sub) > 0:
            master_sub = np.argmax(nres_at_sub)
        
        # Re-define master having the longest length -- forget "j==0" (original designation by estogram2sub)
        sub_redef = [-1 for SS in SSs_all] # -1 == vroot
        for isub,iSSs in enumerate(iSSs_at_sub):
            if len(iSSs) <= 1: continue # treat as regular
            imaster = iSSs[np.argmax([SSs_all[iSS].nres for iSS in iSSs])]
            for iSS in iSSs:
                if iSS == imaster:
                    if isub == master_sub:
                        sub_redef[iSS] = -2 # master of sub0 -- freeze
                    else:
                        sub_redef[iSS] = -1
                else:
                    sub_redef[iSS] = imaster #rest as branch
            
        ## 2. Define jumps: scan through SS
        jumps = []
        for iSS,SS in enumerate(SSs_all):
            j = sub_redef[iSS] #master iSS index
            if j < 0: anc = self.nres+1             #MasterChunk
            else:     anc = SSs_all[j].cenres #BranchChunk
            movable = (anc == self.nres+1) and (j==-1) #unused here....
            jump = Jump(anc,SS.cenres,SS.reslist,movable=movable,is_ULR=(iSS >= n_nonulrSS))

            extra = ""
            if movable:
                extra = " Movable"
            elif j == -2: extra = " Master"
            elif anc != self.nres+1:
                if len(iSSs_at_sub) > 0 and (iSS in iSSs_at_sub[0]):
                    extra = " master-dependent"
                else:
                    extra = " movable-dependent"

            Qjump = np.mean(self.Qres[jump.reslist[0]-1:jump.reslist[1]])
            print("Jump %d (%3d-%3d), %3d-> %3d, Qconf %5.3f %s"%(len(jumps),SS.begin,SS.end,
                                                                  SS.cenres,anc,Qjump,
                                                                  extra))
            jump.ijump = len(jumps) #necessary?
            jumps.append(jump)

        ## 3. Find cut points b/w jumps 
        jumpres = [jump.reslist for jump in jumps]
        jumpres.sort()
        if self.cuttype == 'mid':
            cuts = []
            for i,reslist in enumerate(jumpres[:-1]):
                i1 = jumpres[i][-1]
                i2 = jumpres[i+1][0]
                cuts.append(int((i2+i1)/2))
        else:
            cuts = [jumpres[i+1][0] for i,reslist in enumerate(jumpres[:-1])]
            
        cuts.append(self.nres)
        return cuts,jumps
        
    ## UNUSED: manual
    def setup_fold_tree_from_defined_subs(self,pose,subs):
        ft = pose.conformation().fold_tree().clone()
        self.nres = pose.size()
        if pose.residue(self.nres).is_virtual_residue():
            self.nres -= 1

        jumps = []
        cuts = []
        for isub,sub in enumerate(subs):
            for i,chunk in enumerate(sub):
                chunkcen = get_COMres(pose,chunk)
                cuts.append(chunk[-1])
                if i == 0:
                    ancres = chunkcen
                    # anc,cen,range,movable
                    jumps.append(Jump(self.nres+1,chunkcen,chunk,isub>0)) #place jump at central res to vroot
                else:
                    jumps.append(Jump(ancres,chunkcen,chunk,True)) #place jump at central res to sub-anchor
                
        cuts.append(self.nres)
    
        stat = tree_from_jumps_and_cuts(ft,self.nres+1,jumps,cuts,self.nres+1)
        
        if self.nres == pose.size():
            rosetta.core.pose.addVirtualResAsRoot(pose)
    
        rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        rosetta.protocols.loops.add_cutpoint_variants( pose )
        return jumps

    def save_as_npz(self,outf,
                    Qcut_for_movable_jump=-1.0,fQcut=-1.0):

        nSS = len(self.SSs_reg+self.SSs_ulr)
        jumpinfo = np.ndarray((nSS,7),dtype=int) #begin,end,cut,cen,anc,is_ulr
        nres =  self.cuts[-1] #safe?

        # get estimation of Q at jumpSSs
        jumpmove_by_Q = False
        if fQcut != -1.0 and Qcut_for_movable_jump < 0:
            jumpmove_by_Q = True
            Q_SS = []
            for jump in self.jumps: Q_SS += list(self.Qres[jump.reslist[0]-1:jump.reslist[-1]])
            Q_SS.sort()
            
            Qcut_for_movable_jump = Q_SS[int(fQcut*nres)] #override
            print("fQcut %.3f -> define %.3f as Qcut (mean(jumpQ) below will be movable jump"%(fQcut,Qcut_for_movable_jump))
        
        # 1. residue indices defining jump & cut
        for i,SS in enumerate(self.SSs_reg):
            jump = self.jumps[i] # SS<->jump match?
            Qjump = np.mean(self.Qres[jump.reslist[0]-1:jump.reslist[-1]])
            # by default don't move
            if jumpmove_by_Q:
                movable = (jump.anchor==nres+1) and (Qjump<Qcut_for_movable_jump)
            else:
                movable = jump.movable
            # override as "off" if extended & allow_beta_jump off
            if not self.allow_beta_jump and SS.SStype == 'E':
                movable  = False

            jumpinfo[i] = np.array([SS.begin,SS.end,self.cuts[i],jump.cen,jump.anchor,
                                    0,movable],dtype=int)
            
        # candidates of ULR & vroot stubs
        # simplify for now -- share a common
        vrootstub = self.stub_anc_np
        print("Vroot at saving: \n", vrootstub)

        ulrstubs = [{} for i in range(nSS)]
        nstubs = 0
        for i,SS in enumerate(self.SSs_ulr):
            ij = i+len(self.SSs_reg)
            jump = self.jumps[ij]
            jumpinfo[ij] = np.array([SS.begin,SS.end,self.cuts[ij],jump.cen,jump.anchor,
                                     1,1],dtype=int)

            if jump.stub_anc_np != []:
                ulrstubs[ij]['ulr']    = [moveset.stub_SS_np for moveset in jump.movesets]
                ulrstubs[ij]['tors']    = [moveset.tors for moveset in jump.movesets]
                nstubs += len(ulrstubs[ij]['ulr'])

        print("Movable jumps: ", [i for i,jump in enumerate(jumpinfo) if jump[6]])
        for i,ulrstub in enumerate(ulrstubs):
            if 'ulr' in ulrstub:
                print("Pre-sampled ulr stubs at %d: %d"%(i,len(ulrstub['ulr'])))

        np.savez(outf,
                 Qres=self.Qres,
                 Qres_corr=self.Qres_corr,
                 Qtors=self.Qtors,
                 maxdev=self.maxdev,
                 jump=jumpinfo,
                 ulrs=self.ulrs,
                 vrootstub=vrootstub,
                 ulrstubs=ulrstubs,
                 cuttype=self.cuttype,
                 allow_pickle=True)

    def estimate_torsion_confidence(self,poseinfo,SSpred_fn):
        SS3pred = rosetta_utils.SS9p_to_SS3type(np.load(opt.SSpred_fn)['ss9'])
        torspred = np.load(opt.SSpred_fn)['tors']
        self.Qtors = np.zeros(poseinfo.nres)
        
        # measure by SSmatch & torsion pred entropy
        # range b/w 0~1 in probability scale
        P_SS = np.zeros(poseinfo.nres)
        P_tors = np.zeros(poseinfo.nres)
        for i,SS3 in enumerate(poseinfo.SS3_naive):
            if SS3 == SS3pred[i]: P_SS[i] = 1.0
            else: P_SS[i] = 0.0
            #if SS3 == 'H' or SS3pred[i] == 'H': P_SS = 0.0
            #    else: P_SS = 0.2 # E<->C mismatch
            
            Sphi = myutils.ShannonEntropy(torspred[i][:36])
            Spsi = myutils.ShannonEntropy(torspred[i][36:72])
            P_tors[i] = 1.0/(Sphi*Spsi)

        # renormalize P_tors against most confident residue
        P_tors /= max(P_tors)
        self.Qtors = max(P_tors,P_SS) # better way???

    def load_from_npz(self,npz,pose,
                      randomize_ulrjump=True,
                      report=True,debug=False):
        # Info to fill in
        info = np.load(npz,allow_pickle=True)
        self.Qres = info['Qres']
        self.Qres_corr = info['Qres_corr']
        self.ulrs = info['ulrs']
        self.SSs_ulr = []
        self.jumps = []
        self.cuts = []
        self.cuttype = info['cuttype']

        if pose == None:
            self.nres =  len(self.Qres)
        else:
            if pose.residue(pose.size()).is_virtual_residue(): 
                #PR.rosetta.core.pose.addVirtualResAsRoot(pose)
                pose.delete_residue_slow(pose.size())
            self.nres = pose.size()
            
        if self.nres != len(self.Qres):
            sys.exit("Failed to retrieve FTinfo from file %s: nres inconsistent!"%npz)
            
        if report:
            for i,ulr in enumerate(self.ulrs): print(" - ULR %2d: %3d - %3d"%(i,ulr[0],ulr[-1]))
        
        # first, load jumps and cuts
        self.cuts = []
        self.jumps = []
        for i,[b,e,cut,cen,anc,is_ulr,movable] in enumerate(info['jump']):

            if is_ulr:
                # define as "jump" only if TERMlib defined there
                if info['ulrstubs'][i] == {}:
                    continue
                # randomize ulr jump
                if randomize_ulrjump:
                    ranno = 3.0*random.random()
                    if ranno > 1.0 and cen+2 <= e:
                        cen += 2
                    elif ranno > 2.0 and cen-2 >= b:
                        cen -= 2

            Qjump = np.mean(self.Qres[b-1:e])
            jump = Jump(anc,cen,range(b,e+1),movable=movable,
                        is_ULR=is_ulr,ijump=len(self.jumps),
                        Q=Qjump)
            self.jumps.append(jump)

        # redefine cuts considering ULRs
        alres = [(jump.reslist[0],jump.reslist[-1]) for jump in self.jumps]
        alres.sort()
        for i,(b,_) in enumerate(alres[1:]):
            e = alres[i][1] #previous end
            if self.cuttype == 'mid':
                cut = int((b+e)*0.5)
            else: #end
                cut = b-1 #curr b-1
            self.cuts.append(cut)
        self.cuts.append(self.nres)
            
        if report:
            for i,jump in enumerate(self.jumps):
                a = max([1]+[cut+1 for cut in self.cuts if cut < jump.cen])
                b = min([]+[cut for cut in self.cuts if cut > jump.cen])
                form = " - Jump %2d (%3d-%3d), %3d -> %3d, is_ULR/move=%d/%d, Q=%5.3f"
                print(form%(i,a,b,jump.cen,jump.anchor,jump.is_ULR,jump.movable,jump.Q))
            
        if report: print(" - Cuts: ", self.cuts )

        if pose == None: return
        
        ft = pose.conformation().fold_tree().clone()
        jumpdef = [(jump.anchor,jump.cen) for jump in self.jumps]
        stat = rosetta_utils.tree_from_jumps_and_cuts(ft,self.nres+1,jumpdef,
                                                      self.cuts,self.nres+1)
        if not stat:
            print("Failed to define FoldTree from file %s, return!"%(npz))
            return False

        if not pose.residue(pose.size()).is_virtual_residue():
            PR.rosetta.core.pose.addVirtualResAsRoot(pose)
            
        PR.rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
        PR.rosetta.protocols.loops.add_cutpoint_variants( pose )

        # (optional) load stub set if exists
        # load placeholders first
        vrootstub = info['vrootstub']

        xyz1  = pose.residue(pose.size()).xyz("ORIG")
        xyz2  = pose.residue(pose.size()).xyz("X")
        xyz3  = pose.residue(pose.size()).xyz("Y")
        id1 = PR.rosetta.core.id.AtomID(1,pose.size()) #ORIG
        id2 = PR.rosetta.core.id.AtomID(2,pose.size()) #X
        id3 = PR.rosetta.core.id.AtomID(3,pose.size()) #Y
        for k in range(3):
            xyz1[k] = vrootstub[0,k]
            xyz2[k] = vrootstub[1,k]
            xyz3[k] = vrootstub[2,k]
        pose.set_xyz( id1, xyz1 )
        pose.set_xyz( id2, xyz2 )
        pose.set_xyz( id3, xyz3 )

        # reinstance so that it doesn't mess up vroot
        xyz1 = PR.rosetta.numeric.xyzVector_double_t(0.0)
        xyz2 = PR.rosetta.numeric.xyzVector_double_t(0.0)
        xyz3 = PR.rosetta.numeric.xyzVector_double_t(0.0)
        
        for i,jump in enumerate(self.jumps):
            if info['ulrstubs'][i] == {}: continue

            # get stubs
            for j,stub_np in enumerate(info['ulrstubs'][i]['ulr']):
                moveset = JumpMove()
                moveset.tors = info['ulrstubs'][i]['tors'][j]
                for k in range(3):
                    xyz1[k] = stub_np[0,k] #CA
                    xyz2[k] = stub_np[1,k] #N
                    xyz3[k] = stub_np[2,k] #Cprv
                moveset.stub_SS = PR.rosetta.core.kinematics.Stub(xyz1,xyz2,xyz3)
                jump.movesets.append(moveset)

        # Validate
        # foldtree by randomly perturbing loop residues
        if debug:
            SSres = []
            for jump in self.jumps:
                SSres += jump.reslist
            loopres = [res for res in range(1,self.nres+1) if res not in SSres]

            pertres = []
            for k in range(10):
                ires = random.choice(loopres)
                pose.set_phi(ires,-120)
                pose.set_psi(ires, 120)
                pertres.append(ires)
            pose.dump_pdb("validft.pdb")

        return True
    
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
        
if __name__ == "__main__":
    npz = sys.argv[1]
    FTInfo = FoldTreeInfo(pose=None,opt=None)
    FTInfo.load_from_npz(npz,pose=None,report=True)
