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
        fres = loops[i+1][0]-1
        if fres-ires+1 >= min(min_chunk_len):
            SSs.append(list(range(ires,fres+1)))
    return SSs, [SS3stat[SS[0]-1] for SS in SSs] #SS3stat

def extraSS_from_prediction(regions,SS3pred,report=True):
    extraSSs = []
    SStypes = []
    for reg in regions:
        SS3parts = myutils.list2part([SS3pred[res-1] for res in reg])
        ni = reg[0]
        print(reg,SS3parts)
        for i,part in enumerate(SS3parts):
            SStype = part[0]
            
            # ignore if same SS as reg-stem
            if (i == 0 and reg[0] != 1 and SStype == SS3pred[reg[0]-2]) or \
               (i == len(SS3parts)-1 and reg[-1] != max(SS3pred) and part[-1] == SS3pred[reg[-1]]):
                ni += len(part)
                continue
                
            if (SStype == 'E' and len(part) >= 3) or (SStype == 'H' and len(part) >= 7):
                extraSSs.append( list(range(ni,ni+len(part))) )
                SStypes.append(SStype)
                if report: print("SS-ULR fromPred: %3d-%3d, type=%s"%(ni,ni+len(part),SStype))
            ni += len(part)
    return extraSSs, SStypes

def SS9p_to_SS3type(SS9p):
    SS3type = []
    for p in SS9p:
        pSS3 = [sum(p[0:3]), sum(p[3:5]), sum(p[5:-1])]
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
                             jumps, # vector(pair)
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

def SS_to_jump( pose, SS, ancres, FTInfo, change_pose=False ):
    # ancmap: where in pose does SS.refpos matches

    # 1. find relevant jump id 
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

    for k in range(3):
        xyz_n[k]  = SS.bbcrds_al[SS.cenpos][0][k]
        xyz_ca[k] = SS.bbcrds_al[SS.cenpos][1][k]
        xyz_cp[k] = SS.bbcrds_al[SS.cenpos-1][2][k]

    # 3. Define jump from stubs
    vroot = pose.residue(pose.size())
    stub_anc = rosetta.core.kinematics.Stub(xyz_ca,xyz_n,xyz_cp)
    stub_vroot = rosetta.core.kinematics.Stub(vroot.xyz("ORIG"),vroot.xyz("X"),vroot.xyz("Y"))
    newjump = rosetta.core.kinematics.Jump(stub_vroot,stub_anc)

    # replace jump for validation
    if change_pose:
        ancID = rosetta.core.id.AtomID(2,ancres)
        pose.conformation().set_jump(ijump+1,newjump) #+1: 1-indexing
    return newjump
    
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

