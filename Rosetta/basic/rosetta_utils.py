import sys,copy
import numpy as np
from pyrosetta import *
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/../../utils'%SCRIPTDIR)
import myutils

def relax(pose,sfxn,short=True):
    mm_full = MoveMap()
    mm_full.set_bb(True)
    mm_full.set_chi(True)

    relax = rosetta.protocols.relax.FastRelax(sfxn)
    relax.set_movemap(mm_full)
    # script lines
    lines = rosetta.std.vector_std_string()
    lines.append("switch:torsion")
    lines.append("repeat 2")
    lines.append("ramp_repack_min 0.02  0.01     1.0  50")
    lines.append("ramp_repack_min 0.250 0.01     0.5  50")
    lines.append("ramp_repack_min 0.550 0.01     0.0 100")
    lines.append("ramp_repack_min 1     0.00001  0.0 200")
    lines.append("accept_to_best")
    lines.append("endrepeat")
    lines.append("switch:cartesian")
    lines.append("repeat 1")
    #lines.append("ramp_repack_min 0.02  0.01     1.0  50")
    lines.append("ramp_repack_min 0.250 0.01     0.5  50")
    #lines.append("ramp_repack_min 0.550 0.01     0.0 100")
    lines.append("ramp_repack_min 1     0.00001  0.0 200")
    lines.append("accept_to_best")
    lines.append("endrepeat")
    relax.set_script_from_lines(lines)    
    relax.apply(pose)
    
def pose2SSs(pose,maskres=[],min_chunk_len=[3,5,1000], #[E,H,C]; C: preserve unmasked res if whole looplen > x
             min_loop_len=3,report=False): #default param in hybridize

    pose_work = pose.clone()
    #print( "mask", maskres )
    for res in maskres:
        #unmasked.remove(res)
        for atm in range(1,pose_work.residue(res).natoms()+1):
            atmid = rosetta.core.id.AtomID(atm,res)
            xyz = pose_work.xyz(atmid)
            xyz[0] += 999.0
            xyz[1] += 999.0
            xyz[2] += 999.0
            pose_work.set_xyz( atmid, xyz )
    
    dssp = rosetta.core.scoring.dssp.Dssp(pose_work)

    #SS3stat = [a for a in dssp.get_dssp_secstruct()] # preseve IG -- hairpin treated as one
    SS3stat = [a for a in dssp.get_dssp_reduced_IG_as_L_secstruct()] # as a string
    for res in maskres: SS3stat[res-1] = 'U' #== ULR
    Hs = []
    Es = []
    for i,res in enumerate(range(1,pose.size()+1)):
        if pose.residue(res).is_virtual_residue(): continue #always Coil
        if SS3stat[i] == 'H':
            if len(Hs) == 0 or SS3stat[i] != SS3stat[i-1]: Hs.append([])
            Hs[-1].append(res)
        elif SS3stat[i] == 'E':
            if len(Es) == 0 or SS3stat[i] != SS3stat[i-1]: Es.append([])
            Es[-1].append(res)

    # remove shorter-than-3res E
    for E in Es:
        if len(E) < min_chunk_len[0]:
            for res in E: SS3stat[res-1] = 'L'
    # remove shorter-than-5res H
    for H in Hs:
        if len(H) < min_chunk_len[1]:
            for res in H: SS3stat[res-1] = 'L'

    if report:
        SS3stat = ''.join(SS3stat)
        n10 = int(pose.size()/10)
        print( "SS3fromPose: ", "%-10d"*n10%tuple([i*10+1 for i in range(n10)]))
        print( "SS3fromPose: ", SS3stat )

    # define loops
    loops = []
    for i,res in enumerate(range(1,pose.size()+1)):
        if pose.residue(res).is_virtual_residue(): continue #always Coil
        if SS3stat[i] in ['L','U']:
            if len(loops) == 0 or SS3stat[i] != SS3stat[i-1]: loops.append([])
            loops[-1].append(res)

    for i,loop in enumerate(copy.copy(loops)):
        # reduce loop size for the long ones containing unmasked res
        if len(loop) > min_chunk_len[2]:
            for res in copy.copy(loop):
                if res not in maskres: #=="aligned"
                    loop.remove(res)
            loops_split = list2part(loop)
            loops.remove(loop)
            loops += loops_split
    loops.sort()
            
    # original logic in hybridize to connect SSs connected by short loop
    for i,loop in enumerate(copy.copy(loops)):
        if len(loop) < min_loop_len: loops.remove(loop)

    SSs = [] #make a split list
    for i,loop in enumerate(loops):
        if i == 0:
            if loop[0] >= min(min_chunk_len): #n-term
                SSs.append(list(range(1,loop[0])))
        elif i == len(loops)-1: #c-term
            if pose.size()-1-loop[-1] >= min(min_chunk_len):
                SSs.append(list(range(loop[-1],pose.size())))
            continue
        
        ires = loop[-1]+1
        if i == len(loops)-1: fres = pose.size()
        else: fres = loops[i+1][0]-1
        
        if fres-ires+1 >= min(min_chunk_len):
            SSs.append(list(range(ires,fres+1)))
    return SSs, [SS3stat[SS[0]-1] for SS in SSs] #SS3stat -- this not working properly?

def extraSS_from_prediction(regions,SS3pred,mask_in=None,report=True,minlen=5):
    extraSSs = []
    SStypes = []
    for reg in regions:
        SS3parts = myutils.list2part([SS3pred[res-1] for res in reg])
        ni = reg[0]
        for i,part in enumerate(SS3parts):
            SStype = part[0]
            
            # ignore if same SS as reg-stem
            if (i == 0 and reg[0] != 1 and SStype == SS3pred[reg[0]-2]) or \
               (i == len(SS3parts)-1 and reg[-1] != max(SS3pred) and part[-1] == SS3pred[reg[-1]]):
                ni += len(part)
                continue

            SSreg = list(range(ni,ni+len(part)))
            # check if overlaps with already claimed res
            overlap = [res for res in SSreg if res in mask_in] #0-index
            if overlap != []:
                ni += len(part)
                continue
            
            mask = [i+1 for i in range(len(SS3pred)) if (SS3pred[i] in ['H','E'] and i+1 not in reg)]

            if (SStype == 'E' and len(part) >= 3) or (SStype == 'H' and len(part) >= 7):
                # be permissive to get more anchors!
                if len(SSreg) <= minlen:
                    res1,res2 = try_SS_extension_by_prediction(SSreg,SS3pred,mask,SStype,
                                                               allow_coil=True,maxext=2)
                    if res2-res1 >= 4:
                        SSreg = list(range(res1,res2+1))
                        print("extend:", SSreg)

                #skip if cannot be extended to 5-residues (for frame definition) 
                if len(SSreg) >= 5:
                    extraSSs.append( SSreg )
                    SStypes.append(SStype)
                    if report: print("SS-ULR fromPred: %3d-%3d, type=%s"%(SSreg[0],SSreg[-1],SStype))
            ni += len(part)
            
    return extraSSs, SStypes

def try_SS_extension_by_prediction(reg_org,pred,mask,SStype,
                                   allow_coil=False,maxext=3
                                   ):
    # pred is 0-index, others are 1-index -- nasty...
    
    # extend to Nterm
    res1,res2 = (reg_org[0],reg_org[-1])
    for k in range(maxext):
        if (res1-1 <= 0) or (res1-1 in mask): break
        if allow_coil:
            if pred[res1-2] not in ['C',SStype]: break
        elif (pred[res1-2] != SStype): break
        res1 -= 1

    # extend to Cterm
    for k in range(maxext):
        if (res2 >= len(pred)) or (res2+1 in mask): break
        if allow_coil:
            if pred[res2] not in ['C',SStype]: break
        elif (pred[res2] != SStype): break
        res2 += 1
    return res1,res2

def SS9p_to_SS3type(SS9p):
    #(0: B, 1: E, 2: U, 3: G, 4: H, 5: I, 6: S, 7: T, 8: C); U beta-bulge
    SS3type = []
    for p in SS9p:
        pSS3 = [sum(p[0:3]), sum(p[3:6]), sum(p[6:])] #E,H,C;
        if np.argmax(pSS3) == 1:
            SS3type.append('H')
        elif np.argmax(pSS3) == 0:
            SS3type.append('E')
        else:
            SS3type.append('C')
    return SS3type

def get_COMres(pose,reslist):
    xyzs = np.zeros((len(reslist),3))
    for i,res in enumerate(reslist):
        xyz = pose.residue(res).xyz("CA")
        xyzs[i] = [xyz[0],xyz[1],xyz[2]]
    com = np.sum(xyzs,axis=0)/len(reslist)
    ds = []
    for xyz in xyzs:
        ds.append(np.dot(xyz-com,xyz-com))
    return reslist[np.argmin(ds)]

def tree_from_jumps_and_cuts(ft,
                             nres,  # uint
                             jumps, # vector(pair): (anchor(e.g. vroot),cen)
                             cuts,  # vector
                             root_in #uint
):

    new_topology = True
    if ( len(jumps) == 0 ):
        # make a simple tree. this could also have been done by simple_tree()
        ft.clear()
        if ( root_in != 1 ):
            #this is for re-rooting the tree
            ft.add_edge( 1, root_in, 1 )
            ft.add_edge( root_in, nres, 1 )
            ft.reorder(root_in)
        else:
            ft.add_edge( 1, nres, 1 )
        return True

    #for i,jump in enumerate(jumps):
    #   print "Jump #%d from %d to %d"%(i,jump[0],jump[1])

    #make a list of the unique jump_points in increasing order:
    #so we can construct the peptide edges
    vertex_list = []
    is_cut = [False for i in range(nres)]
    for i,jump in enumerate(jumps):
        vertex_list += jump
        
	#runtime_assert( jump_point_in(1,i) < jump_point_in(2,i) )
        cut = cuts[i]
        is_cut[cut] = True
        vertex_list += [cut,cut+1]

    # remove duplication
    vertex_list.sort()
    for i,val in enumerate(copy.copy(vertex_list)):
        while True:
            if vertex_list.count(val) == 1: break
            vertex_list.remove(val)

    ft.clear() #start building the tree, add peptide edges
    
    jump_stop = vertex_list[0]
    if jump_stop > 1: ft.add_edge( 1, jump_stop, -1 )
    
    for i,start in enumerate(vertex_list[:-1]):
        stop = vertex_list[i+1]
        if not(start >= 1 and start < stop and stop <= nres):
            sys.exit("wrong!")
        if not is_cut[start]: ft.add_edge( start, stop, -1 )

    #Add final edge.
    if vertex_list[-1] < nres: ft.add_edge( vertex_list[-1], nres, -1 )
    for i,jump in enumerate(jumps): 
        ft.add_edge( jump[0], jump[1], i+1 )
    
    #now add the edges corresponding to jumps
    reorder_success = ft.reorder(root_in, True)
    if ( not reorder_success ): return False
    
    #ft.show(T)
    #print T.buf()
    return ft.check_fold_tree(),ft

def SS_to_jump( pose, frameSS, FTInfo, threadable ):
    # 1. find relevant jump id
    ancres = frameSS.cenres # dflt anchor, in pose resno
    found = False
    for ijump,jump in enumerate(FTInfo.jumps):
        if ancres in jump.reslist:
            found = True
            break
    if not found:
        print("JUMP NOT FOUND!")
        return -1

    # 2. get list of core:Vector for stub definition
    # get placeholders of core::Vector types
    xyz_n  = pose.residue(ancres).xyz("N") 
    xyz_ca = pose.residue(ancres).xyz("CA") 
    xyz_cp = pose.residue(ancres-1).xyz("C") 

    # 3. Define jump from stubs
    vroot = pose.residue(pose.size())
    stub_vroot = rosetta.core.kinematics.Stub(vroot.xyz("ORIG"),vroot.xyz("X"),vroot.xyz("Y"))
    stub_vroot_np = np.array([[vroot.xyz("ORIG")[0],vroot.xyz("ORIG")[1],vroot.xyz("ORIG")[2]],
                              [vroot.xyz("X")[0]   ,vroot.xyz("X")[1]   ,vroot.xyz("X")[2]],
                              [vroot.xyz("Y")[0]   ,vroot.xyz("Y")[1]   ,vroot.xyz("Y")[2]]])

    newjumps = []
    stubs = []
    for cenpos in threadable:
        for k in range(3):
            xyz_n[k]  = frameSS.bbcrds_al[cenpos][0][k]
            xyz_ca[k] = frameSS.bbcrds_al[cenpos][1][k]
            xyz_cp[k] = frameSS.bbcrds_al[cenpos-1][2][k]
        stub_anc = rosetta.core.kinematics.Stub(xyz_ca,xyz_n,xyz_cp)
        newjump = rosetta.core.kinematics.Jump(stub_vroot,stub_anc)
        newjumps.append(newjump)

        # in numpy format
        stubinfo = stub_vroot_np + np.array([frameSS.bbcrds_al[cenpos][0],frameSS.bbcrds_al[cenpos][1],frameSS.bbcrds_al[cenpos-1][2]])
        #stubinfo = np.concatenate(stub_vroot_np,stub_anc_np)
        stubs.append(stubinfo)
    return newjumps,stubs,ijump+1 #1-index for Rosetta

def pose2dmtrx(pose):
    nres = pose.size()
    d0mtrx = np.zeros((nres,nres))
    for i in range(nres):
        if pose.residue(i+1).has('CB'): atm_i = 'CB'
        else: atm_i = 'CA'
        xyz_i = pose.residue(i+1).xyz(atm_i)

        for j in range(nres):
            if i >= j: continue
            if pose.residue(j+1).has('CB'): atm_j = 'CB'
            else: atm_j = 'CA'
            xyz_j = pose.residue(j+1).xyz(atm_j)
            d0mtrx[i-1][j-1] = xyz_i.distance(xyz_j)
            d0mtrx[j-1][i-1] = d0mtrx[i-1][j-1]
    return d0mtrx

#def local_extend(pose,extended_mask,reslist="auto",stoppoints=[],idealize=False):
def local_extend(nres,extended_mask,reslist="auto",stoppoints=[]):
    # extend to cutpoints
    if isinstance(reslist,list):
        # get upper- & lower- cut point
        cut_in_region = False
        lower,upper = (min(reslist),max(reslist))
        for res in reslist:
            if res in stoppoints:
                cut_in_region = True
                break
            
        if not cut_in_region:
            while True:
                if lower-1 <= 0 or lower-1 in stoppoints: break
                lower -= 1
            if lower != 1: lower = min(reslist) #do not extend towards N- unless N-terminus ULR
                
            while True:
                if upper+1 > nres or upper+1 in stoppoints: break
                upper += 1
            #if upper == pose.size(): upper = max(reslist) # do not extend towards C if C-term
            
        print("Extend %d:%d -> %d:%d"%(min(reslist),max(reslist),lower,upper))
        reslist = list(range(lower,upper))

    for resno in range(lower,upper+1):
        extended_mask[resno-1] = True #0-index

    '''
    # idealize first
    if idealize:
        idealize_pose(pose,reslist)
        
    # extend up to upper and lower points
    for resno in range(lower,upper+1):
        extended_mask[resno-1] = True #0-index
        pose.set_phi(resno,-135.0)
        pose.set_psi(resno, 135.0)
        pose.set_omega(resno,180.0)
    return pose
    '''

def idealize_pose(pose,reslist='auto'):
    poslist = rosetta.utility.vector1_unsigned_long()
    idealize = rosetta.protocols.idealize.IdealizeMover()

    if reslist == 'auto':
        scorefxn=create_score_function('empty')
        scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
        scorefxn.score(pose)
        emap = pose.energies()
        for res in range(1,pose.size()+1):
            cart = emap.residue_total_energy(res)
            if cart > 50: poslist.append(res)
    elif isinstance(reslist,list):
        for res in reslist: poslist.append(res)

    if len(poslist) == 0: return

    idealize.set_pos_list(poslist)
    idealize.apply(pose)

def restrain_omega(pose,reslist,sig=0.1):
    for res in reslist:
        if res == pose.size() or pose.aa(res+1) == rosetta.core.chemical.aa_pro:
            continue
        
        func1 = rosetta.core.scoring.func.CircularHarmonicFunc(3.141592,sig)
        func2 = rosetta.core.scoring.func.CircularHarmonicFunc(0.0,sig)
        id1 = rosetta.core.id.AtomID(2,res)   #CA
        id2 = rosetta.core.id.AtomID(3,res)   #C
        id3 = rosetta.core.id.AtomID(1,res+1) #N-next
        id4 = rosetta.core.id.AtomID(2,res+1) #CA-next
        id5 = rosetta.core.id.AtomID(4,res)   #O
        id6 = rosetta.core.id.AtomID(pose.residue(res+1).atom_index("H"), res+1) #H-next
        #cst1 = rosetta.core.scoring.constraints.DihedralConstraint( id1, id2, id3, id4, func1 )
        #cst2 = rosetta.core.scoring.constraints.DihedralConstraint( id5, id2, id3, id6, func1 )
        cst3 = rosetta.core.scoring.constraints.DihedralConstraint( id1, id2, id3, id6, func2 ) #ca-c-n-h
        cst4 = rosetta.core.scoring.constraints.DihedralConstraint( id5, id2, id3, id4, func2 ) #o-c-n-ca
        #pose.add_constraint( cst1 )
        #pose.add_constraint( cst2 )
        pose.add_constraint( cst3 )
        pose.add_constraint( cst4 )

def R2quat( R ): #input: Rosetta numeric.xyzMatrix
    if R.xx > R.yy and R.xx > R.zz:
        S = np.sqrt( 1.0 + R.xx - R.yy - R.zz ) * 2.0
        Q = np.array([R.zy-R.yz, 0.25*S*S, R.xy+R.yx, R.zx+R.xz])/S
    elif R.yy > R.zz:
        S = np.sqrt( 1.0 + R.yy - R.xx - R.zz ) * 2.0
        Q = np.array([R.xz-R.zx, R.xy+R.yx, 0.25*S*S, R.zy+R.yz])/S
    else:
        S = np.sqrt( 1.0 + R.zz - R.xx - R.yy ) * 2.0
        Q = np.array([R.yx-R.xy, R.xz+R.zx, R.zy+R.yz, 0.25*S*S])/S
    return Q

def quat2R( Q ): #output: Rosetta numeric.xyzMatrix
    Rnew = rosetta.numeric.xyzMatrix_double_t()
    Rnew.xx=1.0-2.0*(Q[2]*Q[2]+Q[3]*Q[3])
    Rnew.xy=    2.0*(Q[1]*Q[2]-Q[3]*Q[0])
    Rnew.xz=    2.0*(Q[1]*Q[3]+Q[2]*Q[0])
    Rnew.yx=    2.0*(Q[1]*Q[2]+Q[3]*Q[0])
    Rnew.yy=1.0-2.0*(Q[1]*Q[1]+Q[3]*Q[3])
    Rnew.yz=    2.0*(Q[2]*Q[3]-Q[1]*Q[0])
    Rnew.zx=    2.0*(Q[1]*Q[3]-Q[2]*Q[0])
    Rnew.zy=    2.0*(Q[2]*Q[3]+Q[1]*Q[0])
    Rnew.zz=1.0-2.0*(Q[1]*Q[1]+Q[2]*Q[2])
    return Rnew
    
def Qmul(Q1,Q2):
    Q = np.zeros(4)
    Q[0] = Q2[0] * Q1[0] - Q2[1] * Q1[1] - Q2[2] * Q1[2] - Q2[3] * Q1[3]
    Q[1] = Q2[0] * Q1[1] + Q2[1] * Q1[0] + Q2[2] * Q1[3] - Q2[3] * Q1[2]
    Q[2] = Q2[0] * Q1[2] - Q2[1] * Q1[3] + Q2[2] * Q1[0] + Q2[3] * Q1[1]
    Q[3] = Q2[0] * Q1[3] + Q2[1] * Q1[2] - Q2[2] * Q1[1] + Q2[3] * Q1[0]
    Q /= np.sqrt(np.dot(Q,Q))
    return Q

def simple_ft_setup(pose,ulrs):
    nres = pose.size()
    
    ulrres = []
    for ulr in ulrs: ulrres += ulr
    print("ULR: ", ulrres )
    
    SSs, SS3type = pose2SSs(pose,ulrres)

    jumpdef = []
    cuts = []
    resno = 0
    for i,SS in enumerate(SSs):
        cen = SS[int(len(SS)/2)]
        jumpdef.append( (nres+1, cen) )
        if i == len(SSs)-1: cuts.append(nres)
        else:
            loopcen = int((SSs[i+1][0]+SS[-1])/2)
            cuts.append(loopcen)

    #jumpdef = [(69,23),(69,46)] #HACK
    print( "CUTS: ", cuts)
    print( "JUMPS: ", jumpdef )
        
    ft = pose.conformation().fold_tree().clone()
    stat = tree_from_jumps_and_cuts(ft,nres+1,jumpdef,cuts,nres+1)

    if not pose.residue(pose.size()).is_virtual_residue(): 
        rosetta.core.pose.addVirtualResAsRoot(pose)
    rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
    rosetta.protocols.loops.add_cutpoint_variants( pose )

    for res in ulrres:
        pose.set_phi(res,-135.0)
        pose.set_psi(res, 135.0)
        pose.set_omega(res, 180.0)

    return [jump[1] for jump in jumpdef], cuts

def reset_fold_tree(pose, nres=-1, saved_ft=None, verbose=False):
    if nres != -1 and pose.size() != nres:
        pose.conformation().delete_residue_range_slow(nres+1, pose.size())
    if saved_ft != None:
        pose.conformation().fold_tree(saved_ft)

    from pyrosetta.rosetta.core.chemical import CUTPOINT_LOWER, CUTPOINT_UPPER
    connected = []
    for i in range(1, pose.size()+1):
        if pose.residue(i).has_variant_type(CUTPOINT_LOWER):
            is_cut = False #force connect if no reference ft provided
            
            if saved_ft != None: #refer to reference ft
                for ic in range(1, pose.fold_tree().num_cutpoint()+1):
                    cutpoint = pose.fold_tree().cutpoint(ic)
                    if i == cutpoint:
                        is_cut = True
                        break
            if not is_cut:
                pose_changed = True
                if verbose: print("detected cut at %d"%(i))
                rosetta.core.pose.remove_variant_type_from_pose_residue(pose, CUTPOINT_LOWER, i)
                if pose.residue(i+1).has_variant_type(CUTPOINT_UPPER):
                    rosetta.core.pose.remove_variant_type_from_pose_residue(pose, CUTPOINT_UPPER, i+1)
                connected += [i,i+1]
                if i+2 < pose.size(): connected.append(i+2)
    return connected

# writing PDB 
def report_pose(pose,tag,extra_score,outsilent,refpose=None,mute=True):
    # dump to silent
    if outsilent != None:
        silentOptions = rosetta.core.io.silent.SilentFileOptions()
        sfd = rosetta.core.io.silent.SilentFileData(silentOptions)
        
        if not mute: print( "Reporting pose to silent %s..."%outsilent )
        ss = rosetta.core.io.silent.BinarySilentStruct(silentOptions,pose)
        ss.set_decoy_tag(tag)
        for (key,val) in extra_score:
            ss.add_energy(key,val) #add additional info

            if refpose != None:
                natseq = rosetta.core.sequence.Sequence( refpose.sequence(),"native",1 ) 
                seq    = rosetta.core.sequence.Sequence( pose.sequence(),"model",1 ) 
                aln = rosetta.core.sequence.align_naive(seq,natseq)
                gdtmm = rosetta.protocols.hybridization.get_gdtmm(refpose,pose,aln)
                ss.add_energy("gdtmm",gdtmm)
            
        sfd.write_silent_struct(ss,outsilent)
    # dump to individual pdb
    else:
        if not mute: print( "Reporting pose to pdb %s..."%tag)
        pose.dump_pdb(tag)
