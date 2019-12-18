import sys,copy,os
from Jump import *
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/../../utils'%SCRIPTPATH)
from myutils import pdb2crd, pdb2res, pdb_in_resrange, list2part, aa3toaa1, SS_fromdssp, SS_frompross, betainfo_fromdssp, get_naccess_rsa
from myutils import distance, inproduct

class Protein:
    def __init__(self,pdb,ULRs=[],permissive_SSdef=False,report_partial_pdb=False):
        self.pdb = pdb
        self.name = pdb.split('/')[-1].replace('.pdb','')
        self.SSs = []
        self.rsa = []
        self.ULRs = []
        self.res_iSS = {} # SSindex of given resno
        
        self.read_pdb(pdb)
        self.permissive_SSdef = permissive_SSdef

        self.assign_SS_for_ULR(ULRs)

        if report_partial_pdb:
            self.partialpdb = pdb.replace('.pdb','.cpartial.pdb')
            ulrres = []
            for ulr in ULRs: ulrres += ulr
            pdb_in_resrange(pdb,self.partialpdb,ulrres,exres=True)
        
        for ulrdef in ULRs:
            ulrseg = self.define_SS_from_pdb(ulrdef,'C')
            self.ULRs.append(ulrseg)
            
        self.define_SSseg()
        
        #clash check stuffs
        self.clash_grid = []
        self.grid_ix = 0
        self.grid_iy = 0
        self.grid_iz = 0
        self.gridbin = 3.0
        #self.clash_cut = 1

    def read_pdb(self,pdbf):
        self.CAcrd = pdb2crd(pdbf,'CA')
        self.CBcrd = pdb2crd(pdbf,'CB')
        self.bbcrd = pdb2crd(pdbf,'mc')
        self.aas = pdb2res(pdbf)
        self.reslist = self.CAcrd.keys()
        self.reslist.sort()

        self.read_SS_from_pdb(pdbf)

        
    def read_SS_from_pdb(self,pdbf):
        SS3_dssp = SS_fromdssp(pdbf)[1]
        SS3_pross = SS_frompross(self.pdb)
        if SS3_dssp == {}:
            sys.exit('Failed to get SS3_dssp for pdb: %s'%pdbf)
        #print SS3_dssp
        #print SS3_pross

        SS3 = {}
        for res in SS3_dssp:
            if SS3_pross[res] == 'E' or SS3_dssp[res] == 'E':
                SS3[res] = 'E'
            else:
                SS3[res] = SS3_dssp[res]
        self.SS3 = SS3
        self.SS3_naive = copy.deepcopy(self.SS3) #

    def assign_SS_for_ULR(self,ULRs):
        for ulr in ULRs:
            for res in ulr:
                self.SS3[res] = 'C'
        
    def define_SS_from_pdb(self,reg,SStype='',get_crd_al=False):
        CAcrds = []
        CBcrds = []
        bbcrds = []
        seq = ''
        counts = [['H',0],['E',0],['C',0]]
        for res in reg:
            CAcrds.append(self.CAcrd[res])
            CBcrds.append(self.CBcrd[res])
            bbcrds.append(self.bbcrd[res])
            seq += aa3toaa1(self.aas[res])
            iSS = ['H','E','C'].index(self.SS3[res])
            counts[iSS][1] += 1

        if SStype=='':
            counts.sort()
            SStype = counts[-1][0]
        
        SSseg = SSclass(self.name,seq,SStype,reg[0],reg[-1])
        SSseg.CAcrds = CAcrds
        SSseg.CBcrds = CBcrds
        SSseg.bbcrds = bbcrds
        if get_crd_al:
            SSseg.bbcrds_al = copy.deepcopy(SSseg.bbcrds)
        SSseg.get_axes()
        SSseg.get_frame()
        return SSseg
    
    def define_SSseg(self):
        ulrres = []
        for ulr in self.ULRs:
            ulrres += ulr.reslist
            
        self.SSs = []

        SSlist = [self.SS3[res] for res in self.reslist]
        SSparts = list2part(SSlist)

        SSregs = []
        SSs_tmp = []
        SSs = []
        i1 = 1
        for part in SSparts:
            if part[0] != 'C':
                SSregs.append(range(i1,i1+len(part)))
                SSs_tmp.append(part[0])
            i1 += len(part)

        SSs = []
        for i,reg in enumerate(copy.copy(SSregs)):
            if len(reg) < 3:
                SSregs.remove(reg)
            else:
                SSs.append(SSs_tmp[i])
        
        # make sure length is at least 5 & odd
        SSmask = []
        for ireg,reg in enumerate(SSregs):
            ntrial = 0
            res1 = min(reg)
            res2 = max(reg)
            while res2-res1 < 4:
                if ntrial%2 == 0:
                    if res2+1 in self.reslist and res2+1 not in ulrres:
                        res2 +=1
                else:
                    if res1-1 in self.reslist and res1-1 not in ulrres:
                        res1 -=1
                ntrial+=1
            SSregs[ireg] = range(res1,res2+1)
            SSmask += range(res1,res2+1)

        # finally try extension
        for ireg,reg in enumerate(SSregs):
            SSseg_org = self.define_SS_from_pdb(reg,SSs[ireg])

            # extend 
            if SSs[ireg] == 'E' and self.permissive_SSdef:
                SSseg = self.try_beta_extension(SSseg_org,SSmask)
            else:
                SSseg = SSseg_org
            
            for res in SSseg.reslist:
                self.res_iSS[res] = SSseg.iSS
                if res not in SSmask: SSmask.append(res)
                
            SSseg.iSS = len(self.SSs)
            print( 'SSdef: ', ireg, SSseg.SS, SSseg.reslist )
            self.SSs.append(SSseg)


    def define_ULRs(self,ULRdefs):
        self.ULRs = []
        for ulrdef in ULRdefs:
            ulrseg = self.define_SS_from_pdb(ulrdef,'C')
            self.ULRs.append(ulrseg)

    def construct_clash_grid(self):
        ulrres = []
        for ulr in self.ULRs: ulrres += ulr.reslist

        CAcrds_reliable = []
        for res in self.reslist:
            if res in ulrres: continue
            CAcrds_reliable.append(self.CAcrd[res])

        xs = [crd[0] for crd in CAcrds_reliable]
        ys = [crd[1] for crd in CAcrds_reliable]
        zs = [crd[2] for crd in CAcrds_reliable]

        x1,x2 = int(min(xs)/self.gridbin),int(max(xs)/self.gridbin)
        y1,y2 = int(min(ys)/self.gridbin),int(max(ys)/self.gridbin)
        z1,z2 = int(min(zs)/self.gridbin),int(max(zs)/self.gridbin)

        self.grid_ix = x1
        self.grid_iy = y1
        self.grid_iz = z1
        
        nx = x2-x1
        ny = y2-y1
        nz = z2-z1

        self.clash_grid = [[[False for x in range(nz)] for y in range(ny)] for z in range(nx)]

        for crd in CAcrds_reliable:
            ix = int(crd[0]/self.gridbin) - self.grid_ix
            iy = int(crd[1]/self.gridbin) - self.grid_iy
            iz = int(crd[2]/self.gridbin) - self.grid_iz
            if ix >= nx or iy >= ny or iz >= nz: continue
            if ix < 0 or iy < 0 or iz < 0: continue
            self.clash_grid[ix][iy][iz] = True
        
    def SS_is_clashing(self,SS,take_aligned=True,clash_cut=1):
        if self.clash_grid == []:
            self.construct_clash_grid()

        clash_counts = 0
        if take_aligned:
            crds = SS.CAcrds_al
        else:
            crds = SS.CAcrds

        nx = len(self.clash_grid)
        ny = len(self.clash_grid[0])
        nz = len(self.clash_grid[0][0])
        
        for crd in crds:
            ix = int(crd[0]/self.gridbin) - self.grid_ix
            iy = int(crd[1]/self.gridbin) - self.grid_iy
            iz = int(crd[2]/self.gridbin) - self.grid_iz

            if ix >= nx or iy >= ny or iz >= nz: continue
            if ix < 0 or iy < 0 or iz < 0: continue
            
            if self.clash_grid[ix][iy][iz]:
                clash_counts += 1
            if clash_counts >= clash_cut:
                return True
        return False

    def try_beta_extension(self,SSseg,SSmask):
        reg = SSseg.reslist
        res1 = SSseg.begin
        res2 = SSseg.end
        com,z,x = SSseg.axes

        #dx: deviation orthogonal to z-axis
        while True:
            if res1-1 not in self.reslist or res1-1 in SSmask: break
            dv = [com[k]-self.CAcrd[res1-1][k] for k in range(3)]
            dz = inproduct(z,dv)
            dx = distance(dv,[z[k]*dz for k in range(3)])
            v = [self.CAcrd[res1][k]-self.CAcrd[res1-1][k] for k in range(3)]
            normalize(v)
            if inproduct(v,z) < 0.3 or dx > 5.0: break
            res1-=1

        while True:
            if res2+1 not in self.reslist or res2+1 in SSmask: break
            dv = [self.CAcrd[res2+1][k]-com[k] for k in range(3)]
            dz = inproduct(z,dv)
            dx = distance(dv,[z[k]*dz for k in range(3)])
            v = [self.CAcrd[res2+1][k]-self.CAcrd[res2][k] for k in range(3)]
            normalize(v)
            if inproduct(v,z) < 0.3 or dx > 5.0: break
            res2+=1
            
        newreg = range(res1,res2+1)
        return self.define_SS_from_pdb(newreg,SSseg.SS)
    
    def find_strand_pairs(self,find_unpaired=False):
        # just to get bb paired residues from dssp
        strands_by_dssp,dumm1,paired_res,dumm2 = betainfo_fromdssp(self.pdb) 

        # get list of extended definitions of strands
        strands = []
        for iseg,SSseg in enumerate(self.SSs):
            if SSseg.SS == 'E':
                strands.append(SSseg)

                # also store respair info -- ONLY FROM BETA-PAIRING
                paired_res_for_strand = {}
                for ires,res1 in enumerate(SSseg.reslist):
                    paired_res_for_strand[res1] = []
                    for res2 in paired_res[res1]:
                        if res2 in self.res_iSS and self.SSs[self.res_iSS[res2]].SS == 'E':
                            paired_res_for_strand[res1].append(res2)
                self.SSs[iseg].paired_res = paired_res_for_strand
                
        # get extended definitions of paired strands
        strand_pairs = {}
        for i,strand1 in enumerate(strands):
            strand_pairs[i] = []

        n_per_pair = [[0 for i in strands] for j in strands]
        for i,strand1 in enumerate(strands):
            for j,strand2 in enumerate(strands):
                paired_mask = []
                for res1 in strand1.reslist:
                    if res1 not in paired_res: continue
                    for res2 in paired_res[res1]:
                        if res2 not in strand2.reslist: continue
                        paired_mask.append(res1)

                if paired_mask == []: continue
                kmed = len(strand1.reslist)/2
                cenres = strand1.reslist[kmed]
                med_pairedres = sum(paired_mask)*1.0/len(paired_mask)
                q_paired = (med_pairedres-cenres)*2.0/len(strand1.reslist)
                #print strand1.iSS, strand2.iSS, cenres, med_pairedres, q_paired
                if q_paired <= 0.33:
                    n_per_pair[i][j] = len(paired_mask)

        for i,strand1 in enumerate(strands):
            for j,strand2 in enumerate(strands):
                if n_per_pair[i][j] >= 2:
                    strand_pairs[i].append(j)

        one = []
        two = []
        for i,strand in enumerate(strands):
            js = []
            if find_unpaired and len(strand_pairs[i]) <= 1:
                one.append(strands[i])
                if len(strand_pairs[i]) == 1:
                    js = [strand_pairs[i][0]]
            elif not find_unpaired:
                one.append(strands[i])
                if len(strand_pairs) > 0:
                    js = strand_pairs[i]
            else:
                continue
            
            for j in js:
                strandpair = Jump([strands[i],strands[j]])
                strandpair.find_registering_respairs()#seed=paired_res)
                print( 'reg for ', strands[i].iSS, strands[j].iSS, strandpair.register )
                two.append(strandpair)
                
        return one, two

    def find_exposed_SSs(self,exposure_def,SStype):
        # perhaps should do on partial.pdb?
        if self.rsa == []:
            self.rsa = get_naccess_rsa(self.partialpdb)
            for res in self.reslist:
                if res not in self.rsa: self.rsa[res] = 100.0

        # first go through helices and tag if any hydrophobic residue is exposed
        HPaas = ['PHE','ILE','LEU','MET','VAL','TRP']

        exposed_SSs = []
        for iSS,SSseg in enumerate(self.SSs):
            if self.SSs[iSS].SS != SStype: continue
            nexposed = 0
            for res in SSseg.reslist:
                if self.aas[res] in HPaas and self.rsa[res] > 10:
                    nexposed += 1
                    self.SSs[iSS].hydrophobics.append(res)
            if nexposed >= exposure_def:
                exposed_SSs.append(iSS)

        return exposed_SSs
        
    def find_exposed_helix(self):
        exposed_helix = self.find_exposed_SSs(exposure_def=2,SStype="H")

        exposed_one = []
        for iSS in exposed_helix:
            exposed_one.append(self.SSs[iSS])
        return exposed_one

    def find_exposed_helices(self):
        exposed_HHs = self.find_exposed_SSs(exposure_def=2,SStype="H")

        exposed_two = []
        for iSS in exposed_HHs:
            for jSS in exposed_HHs:
                if iSS >= jSS: continue
                SS1 = self.SSs[iSS]
                SS2 = self.SSs[jSS]
                if SS1.is_contacting_SS(SS2):
                    helixpair = Jump([SS1,SS2])
                    helixpair.find_registering_respairs()
                    exposed_two.append(helixpair)
                    
        return exposed_two

    def find_exposed_strands(self):
        exposed_strands = self.find_exposed_SSs(exposure_def=1,SStype="E")
        
        one,two = self.find_strand_pairs()

        exposed_two = []
        for jump in two:
            SS1 = jump.SSs[0]
            SS2 = jump.SSs[1]
            # take if any strand is "exposed"
            if SS1.iSS not in exposed_strands and SS2.iSS not in exposed_strands: continue
            exposed_two.append(jump)
        return exposed_two
    
    def ulr_placeable_at_jump(self,jump,ulrres):
        #first check if ULRseq overlapps with jump
        if not self.permissive_SSdef:
            for SS in jump.SSs:
                for res in (SS.begin,SS.end+1):
                    if res in ulrres:
                        return False

        # return True if adjacent
        for SS in jump.SSs:
            for res in ulrres:
                if abs(SS.begin-res) < 3 or abs(SS.end-res) < 3:
                    return True
        
        anccrd1 = []
        anccrd2 = []
        dperres = 3.0
        if ulrres[0]-self.reslist[0] > 3: #ulr not Nterm
            anccrd1 = self.CAcrd[ulrres[0]]
        if self.reslist[-1]-ulrres[-1] > 3: #ulr not Cterm
            anccrd2 = self.CAcrd[ulrres[-1]]

        jumpcom = [0.0,0.0,0.0]
        for SS in jump.SSs:
            SSrefcrd = SS.get_refcrd()
            for k in range(3):
                jumpcom[k] += SSrefcrd[k]/len(jump.SSs)

        #print 'ulr placeable?', ulrres, jumpcom
        #print 'anchorcrds: ', anccrd1, anccrd2
        
        ## better estimation than nres*3.0 ang (takes into account of folding...)??
        if anccrd1 != [] and anccrd2 != []: #ulr as loop
            half = len(ulrres)/2
            d_jump_to_ulranc1 = distance(jumpcom,anccrd1)
            d_jump_to_ulranc2 = distance(jumpcom,anccrd2)

            if d_jump_to_ulranc1 > dperres*half and d_jump_to_ulranc2 > dperres*half:
                return False
            else:
                return True
        else: # terminus case
            if anccrd1 != []:
                anccrd = anccrd1
            else:
                anccrd = anccrd2
            d_jump_to_ulranc = distance(jumpcom,anccrd)
            
            nres_linker = len(ulrres)-3 # rough estimation of max length
            if d_jump_to_ulranc > dperres*nres_linker:
                return False
            else:
                return True

    def make_threads_from_ulr(self,jumptype,out,report_pdb=False):
        # first look if ulr has SS struct
        SSranges = []
        for ulr in self.ULRs:
            SSs = [self.SS3_naive[res] for res in ulr.reslist]
            SSlist = list2part(SSs)

            ishift = 0
            for i,reg in enumerate(SSlist):
                if i > 0: ishift += len(SSlist[i-1])
                if len(reg) < 3: continue

                #allow +-1 coil
                if reg[0] in ['E','H']:
                    SStype = reg[0]
                    i1 = ulr.reslist[ishift]
                    i2 = ulr.reslist[ishift+len(reg)-1]
                    if len(reg) == 3:
                        SSranges.append((i1-1,i2+1,SStype))

                    elif len(reg) == 4:
                        SSranges.append((i1,i2+1,SStype))
                        #SSranges.append((i1-1,i2,SStype))

                    elif len(reg) >= 5:
                        #k = (len(reg)-5)/2
                        for k in range(len(reg)-4):
                            SSranges.append((i1+k,i1+k+4,SStype))
                
        ## SS structs to thread
        contiguous_SSs = []

        for i1,i2,SStype in SSranges:
            if i1 in self.reslist and i2 in self.reslist:
                SSseg = self.define_SS_from_pdb(range(i1,i2+1),SStype,
                                                get_crd_al=True)
                contiguous_SSs.append(SSseg)

        
        l = '- searching %d possible contig SSs: '%len(contiguous_SSs)
        for SSseg in contiguous_SSs:
            l += ' %d-%d'%(SSseg.begin,SSseg.end)
        print(l)
        
        #make jumps by adding first SS (placeholder)
        SS1 = self.SSs[0]
        for SS2 in contiguous_SSs:
            jump = Jump([SS1,SS2])
            jump.index = 'input'
            
            if report_pdb:
                pdbname = 'match.input.%d.%d.pdb'%(SS2.begin,SS2.end)
            else:
                pdbname = ''

            for ulr in self.ULRs:
                if ulr.nres < 9: continue #allow +-2

                # do not use blosum -- assume input threading is wrong
                threads,bestmatch = ulr.thread_match(jump.SSs[-1],seqscorecut=0.0)

                refjump = Jump([SS1,ulr])
                jump.report_chunk_insertion(refjump,1,threads,
                                            out, 0, 
                                            report_as_pdb=pdbname )
                
                print( '-- Match %d-%d-%d.%d: Total %d threads'%(SS2.begin,SS2.end,
                                                                ulr.begin,ulr.end,len(threads)))
            
        
        
                
