import sys,copy,os
import pyrosetta as PR
import numpy as np
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/../../utils'%SCRIPTDIR)
import myutils

## SSClass should be transferrable <=> pyRefine Jump
class SSclass:
    def __init__(self,SStype,begin,end,poseinfo=None,refpos=-1):
        self.tag = ""
        self.seq = ""
        self.SStype = SStype
        
        self.begin = begin
        self.end = end
        self.reslist = range(begin,end+1)
        self.nres = len(self.reslist)

        self.cenpos = int(self.nres/2)
        self.cenres = begin + self.cenpos
        self.refpos = self.cenpos 
        if refpos != -1: self.refpos = refpos
        
        self.iSS = 0 # SSindex (only for SS at protein)
        self.jumpid = 0 # Which jump index it belongs to in FT

        # extra info
        self.hydrophobics = []
        self.exposed = False

        self.axes = None
        self.CAcrds = np.zeros((self.nres,3))
        self.CBcrds = np.zeros((self.nres,3))
        self.bbcrds = np.zeros((self.nres,5,3))

        self.CAcrds_al = np.zeros((self.nres,3))
        #self.CBcrds_al = np.zeros((self.nres,3))
        self.bbcrds_al = np.zeros((self.nres,5,3))
        
        if poseinfo != None:
            self.tag = poseinfo.tag
            self.seq = poseinfo.seq[begin-1:end]
            self.crds_from_pose(poseinfo)

    def info(self): 
        return ' %s:%d-%d'%(self.SStype,self.begin,self.end)
    
    def crds_from_pose(self,poseinfo):
        self.CAcrds = poseinfo.CAcrds[self.begin-1:self.end]
        self.CBcrds = poseinfo.CBcrds[self.begin-1:self.end]
        self.bbcrds = poseinfo.bbcrds[self.begin-1:self.end]
        self.bbcrds_al = None
        #self.CBcrds_al = None
        self.get_frame() #update axes by default

    # calculate from CAcrds
    def get_axes(self):
        if self.bbcrds != [] and self.CAcrds != []:
            com = self.CAcrds[self.cenpos]
            z = np.mean(self.CAcrds[1:]-self.CAcrds[:-1],axis=0)
            z /= np.sqrt(np.dot(z,z))
            x = self.CBcrds[self.cenpos]-com
            self.axes = [com,z,x]

    # definition of 5 contiguous residues to assign TERM
    def get_frame(self,update_axes=True,n=5):
        half = int(n/2)
        if update_axes: self.get_axes()
        self.frame = self.CAcrds[self.refpos-half:self.refpos+half+1]

    def get_refcrd(self):
        return self.CAcrds[self.refpos]

    # direction of SS
    def get_sign(self,SS2):
        match = np.dot(self.axes[1],SS2.axes[1])
        
        if match > 0.5: sgn = 1
        elif match < -0.5: sgn = -1
        else: sgn = 0
        return sgn
            
    def make_subSS(self,i1,i2):
        seq = self.seq[i1:i2+1]
        SS_w = SSclass(self.SStype,self.begin+i1,self.begin+i2)

        SS_w.CAcrds = self.CAcrds[i1:i2+1,:]
        SS_w.CBcrds = self.CBcrds[i1:i2+1,:]
        SS_w.bbcrds = self.bbcrds[i1:i2+1,:,:]
        SS_w.get_frame() # will call get_axes as well
        return SS_w
        
    def is_contacting_SS(self,SS2):
        ncontact = 0
        for crd1 in self.CBcrds:
            for crd2 in SS2.CBcrds:
                dcrd = (crd1-crd2)
                if np.sum(dcrd*dcrd) < 64.0: ncontact += 1
                if ncontact >= 2:
                    return True # 1 is too few
        return False

    def check_exposed(self,rsa,exposure_def):
        HPaas = ['F','I','L','M','V','W']
        for i,res in enumerate(self.reslist):
            if self.seq[i] in HPaas and rsa[res] > 0.1:
                self.hydrophobics.append(i)
        if len(self.hydrophobics) > exposure_def: self.exposed = True

    # use RMSD...
    def close(self,SS2,take_aligned=True,rmsdcut=1.0):
        if self.nres != SS2.nres: return False

        if take_aligned:
            crd1 = self.CAcrds_al
            crd2 = SS2.CAcrds_al
        else:
            crd1 = self.CAcrds
            crd2 = SS2.CAcrds

        msd = (crd1-crd2)*(crd1-crd2)/self.nres

        if msd < rmsdcut*rmsdcut:
            return True
        else:
            return False

class PoseInfo:
    def __init__(self,pose,opt,FTInfo,SSpred=None):
        self.tag = pose.pdb_info().name() ##???
        
        self.res_iSS = {} # SSindex of given resno

        self.read_pose(pose)
        self.SS3_pred = SSpred #can be "None"

        # 1. Collect anchor info
        # segment SSs -- use SSpred info as well for permissive
        self.define_SSseg(opt,FTInfo)

        # store extra information being used later
        # also report SSsegs defined
        if opt.debug: print( "\n[PoseInfo.init] ======SSsegs defined======")
        rsa = PR.rosetta.core.scoring.sasa.rel_per_res_sc_sasa( pose )
        for iSS,SSseg in enumerate(self.SSs):
            SSseg.check_exposed( rsa, opt.config['EXPOSURE_DEF'][SSseg.SStype])
            if opt.debug:
                print( "SSDEF %d: "%iSS, SSseg.info(), "exposed: %d"%SSseg.exposed )

        # get list of direct SSi-SSj contacts and their registers
        self.find_SSpairs(report=opt.debug)

        # 2. Collect ULR-SS info
        self.ULRs = []
        ulrres = []
        for ulr in FTInfo.ulrs:
            self.ULRs.append( SSclass("U",ulr[0],ulr[-1]) )
            ulrres += ulr

        # 3. build clash grids
        self.gridbin = opt.config['CLASH_GRID_BIN']
        self.build_clash_grid(maskres=ulrres)

    def read_pose(self,pose):
        dssp = PR.rosetta.core.scoring.dssp.Dssp(pose)
        dssp.insert_ss_into_pose( pose )

        self.SS3_naive = [pose.secstruct(ires) for ires in range(1,pose.size()+1)]

        ## coordinate
        nres = pose.size()
        if pose.residue(pose.size()).is_virtual_residue(): nres -= 1
        self.bbcrds = np.zeros((nres,5,3)) 
        self.CAcrds = np.zeros((nres,3))
        self.CBcrds = np.zeros((nres,3))
        for i in range(nres):
            Ncrd = pose.residue(i+1).xyz("N")
            CAcrd = pose.residue(i+1).xyz("CA")
            Ccrd = pose.residue(i+1).xyz("C")
            Ocrd = pose.residue(i+1).xyz("O")
            if pose.residue(i+1).has("CB"):
                CBcrd = pose.residue(i+1).xyz("CB")
            else:
                CBcrd = CAcrd
            self.CAcrds[i] = CAcrd
            self.CBcrds[i] = CBcrd
            self.bbcrds[i][0] = Ncrd
            self.bbcrds[i][1] = CAcrd
            self.bbcrds[i][2] = Ccrd
            self.bbcrds[i][3] = Ocrd
            self.bbcrds[i][4] = CBcrd

        ## sequence
        self.seq = pose.sequence()
        self.reslist = range(1,pose.size()+1)
        #write_as_pdb("pose.ca.pdb",self.CAcrds)
     
    def define_SSseg(self,opt,FTInfo):
        self.SSs = []
        ulrres = []
        for ulr in FTInfo.ulrs: ulrres += ulr

        # make non-coil & >= 3res 
        SSparts = myutils.list2part(self.SS3_naive)
        SSregs = []
        SSmask = [] # already claimed
        i1 = 1
        for part in SSparts:
            SStype = part[0]
            if SStype == 'L':
                i1 += len(part)
                continue
            
            reg = range(i1,i1+len(part))
            if (SStype == 'E' and len(reg) < 3) or \
               (SStype == 'H' and len(reg) < 7): continue

            SSseg = SSclass( SStype, reg[0], reg[-1], poseinfo=self )
            SSseg.jumpid = FTInfo.get_jumpid(reg) #TODO

            # Try extension only if SStype==E -- often structure is slightly broken
            if SStype == 'E':
                if self.SS3_pred != None:
                    SSseg_ext,i1 = self.try_extension_by_prediction(SSseg, SSmask+ulrres)
                else:
                    SSseg_ext,i1 = self.try_extension_by_structure(SSseg, SSmask+ulrres)
                
                SSseg_ext.iSS = len(self.SSs)
                self.SSs.append(SSseg_ext)
            else:
                self.SSs.append(SSseg)
                
            #print( 'SSdef: ', self.SSs[-1].SStype, self.SSs[-1].reslist, self.SSs[-1].axes )
            SSmask += reg
            i1 += len(part)

    # Version using prediction
    def try_SS_extension_by_prediction(self,SSseg,mask):
        reg = SSseg.reslist
        
        # extend to Nterm
        res1,res2 = (SSseg.begin,SSseg.end)
        while True:
            if res1-1 not in self.reslist or res1-1 in SSmask or self.SS3_pred[res1] != SSseg.SStype: break
            res1 -= 1

        # extend to Nterm
        res1,res2 = (SSseg.begin,SSseg.end)
        while True:
            if res1-1 not in self.reslist or res1-1 in SSmask or self.SS3_pred[res1] != SSseg.SStype: break
            res1 -= 1
            
    # Version using structure's bb trace
    def try_SS_extension_by_structure(self,SSseg,mask):
        reg = SSseg.reslist
        com,z,x = SSseg.axes
        
        # z: principal axis of SS, dx: deviation orthogonal to z-axis, v: CA trace direction
        # allow extension if
        # i) crd deviation AND ii) trace are not too off from principal axis

        # extend to Nterm
        res1,res2 = (SSseg.begin,SSseg.end)
        while True:
            if res1-1 not in self.reslist or res1-1 in SSmask: break
            dv = com - self.CAcrd[res1-1]
            dz = np.dot(z,dv)
            dx = (dv-z*dz)*(dv-z*dz)
            v = self.CAcrd[res1] - self.CAcrd[res1-1]
            v /= np.sqrt(np.dot(v,v))
            if np.dot_product(v,z) < 0.3 or dx > 5.0: break
            res1-=1

        # extend to Cterm
        while True:
            if res2+1 not in self.reslist or res1+1 in SSmask: break
            dv = self.CAcrd[res2+1] - com
            dz = np.dot(z,dv)
            dx = (dv-z*dz)*(dv-z*dz)
            v = self.CAcrd[res2+1] - self.CAcrd[res2]
            v /= np.sqrt(np.dot(v,v))
            if np.dot_product(v,z) < 0.3 or dx > 5.0: break
            res2 +=1

        return SSclass(SSseg.SStype,res1,res2,poseinfo=self)
    
    # find direct contacting (e.g. sheet-pairing) residues
    def find_SSpairs(self,report=False):
        self.SSpairs = []
        self.register = [{} for SS in self.SSs]
        if report: print( "\n[PoseInfo.find_SSpairs] ======SS pairs defined======")
        
        for iSS,SS1 in enumerate(self.SSs):
            for jSS,SS2 in enumerate(self.SSs):
                if jSS <= iSS: continue

                # EE -- check the axes of SS; skip if not parallel nor antiparallel
                sgn = SS1.get_sign(SS2)
                if SS1.SStype+SS2.SStype == 'EE' and sgn == 0: continue
                
                # respairs < 7 Ang and least deviating from SScenter
                d2map = np.array([[np.dot(SS1.CBcrds[i1] - SS2.CBcrds[i2],SS1.CBcrds[i1] - SS2.CBcrds[i2]) \
                                   for i1 in range(SS1.nres)] for i2 in range(SS2.nres)])
                cmap = d2map<49.0

                # com-to-com vector
                vcom = SS1.axes[0]-SS2.axes[0]
                vcom /= np.sqrt(np.dot(vcom,vcom))

                # should have at least 3 contacts 
                if np.count_nonzero(cmap==1) <3: continue
                
                # should not align on top of each other ( cos(30deg) = 0.866 )
                if abs(np.dot(SS1.axes[1],vcom)) > 0.866 or abs(np.dot(SS2.axes[1],vcom)) > 0.866: continue

                self.SSpairs.append((iSS,jSS))

                # get all pairs directly contacting
                self.register[iSS][jSS] = []
                self.register[jSS][iSS] = []
                for i in range(SS1.nres):
                    for j in range(SS2.nres):
                        if cmap[j][i]:
                            self.register[iSS][jSS].append((i,j))
                            self.register[jSS][iSS].append((j,i))
                if report:
                    print("SSPAIR: %d-%d, cen %d-%d, %d respairs"%(iSS,jSS,SS1.cenres,SS2.cenres,len(self.register[iSS][jSS])))

    # new: preprocess
    def get_possible_1ancframe_given_size(self,wsize,SSs, #this is index!
                                          FTInfo, #should contain estogram or lddt
                                          skip_if_strand_paired_in_both_direcs=True,
                                          include_hydrophobic=False):
        anchorSSs = []
        # 1-anchorSS case
        for iSS in SSs:
            SS = self.SSs[iSS]
            for i in range(SS.nres-wsize+1):
                i1 = i
                i2 = i+wsize-1
                usable = True
                
                # skip if hydrophobic res not included as specified
                if include_hydrophobic:
                    nHP = 0
                    for ires in range(i1,i2+1):
                        resno = SS.begin + ires
                        if resno in SS.hydrophobics: nHP += 1
                    if nHP < include_hydrophobic:
                        usable = False
                
                # skip if the strand already paired in both directions
                if SS.SStype == 'E' and skip_if_strand_paired_in_both_direcs:
                    for ires in range(i1,i2+1):
                        resno = SS.begin + ires
                        if resno in SS.paired_res and len(SS.paired_res[resno]) >= 2:
                            usable = False
                            print( '---  chunk %d-%d not usable as a register because of both-paired res %d'%(SS.begin+i1,SS.begin+i2,resno) )
                            break
                if not usable: continue

                SS_w = SS.make_subSS(i1,i2) #not resno, index no within fullSS
                anchorSSs.append([SS_w])
        return anchorSSs

    # new: preprocess from pose
    def get_possible_2ancframe_given_size(self,wsize,SSpairs, #this is index!
                                          FTInfo, #should contain estogram or lddt
                                          allowed_anchor_SStypes,
                                          minres_for_frame=5,
                                          multiple_regs=False):
        half = int(wsize/2)
        anchorSSs = []
        for (iSS1,iSS2) in SSpairs: #enumerate through pairs
            SS1,SS2 = (self.SSs[iSS1],self.SSs[iSS2])
            
            if (SS1.SStype+SS2.SStype) not in allowed_anchor_SStypes: continue
            
            # consider only having pair registers
            if iSS2 not in self.register[iSS1]: return []
            regs_ij = self.register[iSS1][iSS2]

            for i,j in regs_ij:
                ib,ie = (i-half,i+half)
                jb,je = (j-half,j+half)
                if ib < 0 or jb < 0 or ie >= SS1.nres or je >= SS2.nres: continue

                SS1_w = SS1.make_subSS(ib,ie)
                SS2_w = SS2.make_subSS(jb,je)
                anchorSSs.append([SS1_w,SS2_w])
                
                '''
                shift1 = (i, i+wsize-1) #new ibegin/iend of subSS1
                SS1_w = SS1.make_subSS(shift1[0],shift1[1])

                # register match!
                match = []
                for (i1,i2) in regs_ij: # get range of registers within window 
                    if i1 >= i and i1 <= i+wsize-1: match.append((i1,i2))
                # take consistent subset?
                if len(match) < wsize: continue

                #new ibegin/iend of subSS2
                shift2 = (min(match[0][1],match[-1][1]),max(match[0][1],match[-1][1])) 

                if multiple_regs: ## HH jump
                    for offset in range(-4,5): 
                        (i1,i2) = (shift2[0]+offset,shift2[1]+offset)
                        if i2-i1 < minres_for_frame-1: continue
                        if i1 < 0 or i2 >= SS2.nres: continue

                        SS2_w = SS2.make_subSS(i1,i2)
                        jump_windows.append([SS1_w,SS2_w])
                else:
                    if shift2[1]-shift2[0] < minres_for_frame-1: continue
                    SS2_w = SS2.make_subSS(shift2[0],shift2[1])
                    anchorSSs.append([SS1_w,SS2_w])
                '''
        return anchorSSs

    def get_ULR_anchors(self,FTInfo,anctypes,report=False):
        # 1. get list of not super great SS/SSpair list
        # TODO: filter list using FTInfo.estogram here??
        unpaired_oneE,unpaired_twoE = self.find_unpaired_anchors(['E','EE'],report=report)
        exposed_one,exposed_two     = self.find_exposed_anchors(anctypes,report=report)
        if report:
            print( "\n[PoseInfo.get_ULR_anchors] =======Potential SS-ULR anchors========= ")
            print( "Unpaired1: ", unpaired_oneE)
            print( "Unpaired2: ", unpaired_twoE)
            print( "Exposed1 : ", exposed_one )
            print( "Exposed2 : ", exposed_two )

        # should redundancy removed?
        SS2anchors = unpaired_twoE + exposed_two
        SS1anchors = unpaired_oneE + exposed_one
        
        # 2. Get framelist given SS-anchors
        # "frame" is sub-SS of SS
        frame1 = self.get_possible_1ancframe_given_size(5,SS1anchors,FTInfo)
        #minres_for_frame=minres_for_frame)
        frame2 = self.get_possible_2ancframe_given_size(5,SS2anchors,FTInfo,anctypes)
        #minres_for_frame=minres_for_frame)
        
        # results are 5-res segments
        if report:
            l = "AncFrames,1:"
            for i,SSs in enumerate(frame1):
                l += "%8s;"%SSs[0].info()
            print( l )
            l = "AncFrames,2:"
            for i,SSs in enumerate(frame2):
                for SS in SSs: l += "%8s"%SS.info()
                l += ";"
            print( l )
        return frame1, frame2

    def find_unpaired_anchors(self,allowed_anchor_SStypes,report=False):
        unpaired_one = []
        unpaired_two = []

        # first get num SSpair-of-interest (e.g. EE) 
        npairs = {}
        for (iSS1,iSS2) in self.SSpairs:
            SS1 = self.SSs[iSS1]
            SS2 = self.SSs[iSS2]
            if SS1.SStype+SS2.SStype in allowed_anchor_SStypes:
                npairs[iSS1] += 1
                npairs[iSS2] += 1

        for iSS,SS in enumerate(self.SSs):
            if SS.SStype not in allowed_anchor_SStypes: continue
            if npairs[iSS] < 2: unpaired_one.append(iSS)
                
        for (iSS1,iSS2) in self.SSpairs:
            if SS1.SStype+SS2.SStype not in allowed_anchor_SStypes: continue
            if npairs[iSS1] < 2 or npairs[iSS2] < 2:
                unpaired_two.append((iSS1,iSS2))
                
        return unpaired_one, unpaired_two

    def find_exposed_anchors(self,allowed_anchor_SStypes,report=False):
        exposed_one = []
        exposed_two = []
        for iSS,SS in enumerate(self.SSs):
            if SS.SStype in allowed_anchor_SStypes and SS.exposed:
                exposed_one.append(iSS)
        
        for (iSS1,iSS2) in self.SSpairs:
            SS1 = self.SSs[iSS1]
            SS2 = self.SSs[iSS2]
            if not SS1.exposed or not SS2.exposed: continue
            
            if SS1.SStype+SS2.SStype in allowed_anchor_SStypes:
                exposed_two.append((iSS1,iSS2))
            elif SS2.SStype+SS1.SStype in allowed_anchor_SStypes:
                exposed_two.append((iSS2,iSS1))

        return exposed_one, exposed_two

    def build_clash_grid(self,maskres=[]):
        xyz_reliable = []
        for res in self.reslist:
            if res in maskres or res-1 >= len(self.CAcrds): continue
            xyz_reliable.append(self.CAcrds[res-1])

        xyz_reliable = np.array(xyz_reliable)

        # just store beginning
        self.grid_bxyz = (np.min(xyz_reliable,axis=0)/self.gridbin).astype(int)
        grid_exyz = (np.max(xyz_reliable,axis=0)/self.gridbin).astype(int)

        nxyz = grid_exyz - self.grid_bxyz + 1
        self.clash_grid = np.zeros((nxyz[0],nxyz[1],nxyz[2]))
        for crd in xyz_reliable:
            icrd = (crd/self.gridbin).astype(int) - self.grid_bxyz
            if (icrd<0).any() or ((icrd-nxyz)>0).any(): continue
            self.clash_grid[icrd[0]][icrd[1]][icrd[2]] = 1

        # debug
        '''
        crds_tmp = []
        for i in range(nxyz[0]):
            x = (grid_bxyz[0]+i)*self.gridbin
            for j in range(nxyz[1]):
                y = (grid_bxyz[1]+j)*self.gridbin
                for k in range(nxyz[2]):
                    z = (grid_bxyz[2]+k)*self.gridbin
                    if self.clash_grid[i][j][k]:
                        crds_tmp.append(np.array([x,y,z]))
        myutils.write_as_pdb("grid.pdb",crds_tmp)
        '''
        
    def SS_is_clashing(self,SS,take_aligned=True,clash_cut=1):
        clash_counts = 0
        if take_aligned: crds = SS.CAcrds_al
        else: crds = SS.CAcrds

        nxyz = self.clash_grid.shape
        for crd in crds:
            icrd = (crd/self.gridbin).astype(int) - self.grid_bxyz
            if (icrd<0).any() or ((icrd-nxyz)>=0).any(): continue
            clash_counts += self.clash_grid[icrd[0]][icrd[1]][icrd[2]]
            if clash_counts >= clash_cut:
                return True
            
        return False
