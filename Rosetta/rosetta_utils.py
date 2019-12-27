import sys,copy
import numpy as np
from pyrosetta import *

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
    #unmasked = [res for res in range(1,pose.size()+1)]
    print( maskres )
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
    for res in maskres:
        SS3stat[res-1] = 'L' #== ULR
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
        print( "SS3: ", "%-10d"*n10%tuple([i*10+1 for i in range(n10)]))
        print( "SS3: ", SS3stat )

    # define loops
    loops = []
    for i,res in enumerate(range(1,pose.size()+1)):
        if pose.residue(res).is_virtual_residue(): continue #always Coil
        if SS3stat[i] == 'L':
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
        
    if report: print ("loops",loops)

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
    return SSs,SS3stat

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
    return ft.check_fold_tree()

def setup_fold_tree(pose,ulrs,opt,
                    additional_SS_def=[]):
    #if opt.subs != []:
    #    jumps,movable = setup_fold_tree_from_defined_subs(pose,opt.subs)
    #else:
    jumps,movable = setup_fold_tree_from_SS(pose,ulrs,opt,
                                            additional_SS_def=additional_SS_def,
                                            subs=opt.subs,
                                            report=opt.verbose)
    return jumps,movable

def setup_fold_tree_from_defined_subs(pose,subs):
    ft = pose.conformation().fold_tree().clone()
    nres = pose.size()
    if pose.residue(nres).is_virtual_residue():
        nres -= 1

    jumps = []
    cuts = []
    movable = []
    for sub in subs:
        for i,chunk in enumerate(sub):
            chunkcen = get_COMres(pose,chunk)
            cuts.append(chunk[-1])
            if i == 0:
                ancres = chunkcen
                jumps.append((nres+1,chunkcen)) #place jump at central res to vroot
                movable.append(True)
            else:
                jumps.append((ancres,chunkcen)) #place jump at central res to sub-anchor
                movable.append(False)
                
    cuts.append(nres)
    
    stat = tree_from_jumps_and_cuts(ft,nres+1,jumps,cuts,nres+1)
        
    if nres == pose.size():
        rosetta.core.pose.addVirtualResAsRoot(pose)
    
    rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
    rosetta.protocols.loops.add_cutpoint_variants( pose )
    return jumps, movable

def setup_fold_tree_from_SS(pose,ulrs,opt,
                            min_chunk_len=[3,5,1000],
                            variable_anchor=False,variable_cutpoint=False,
                            additional_SS_def=[],
                            subs=[],
                            report=False):
    ft = pose.conformation().fold_tree().clone()
    nres = pose.size()
    if pose.residue(nres).is_virtual_residue():
        nres -= 1

    ulrres = []
    for ulr in ulrs: ulrres += ulr

    # get SS seginfo from pose
    SSs,SS3stat = pose2SSs(pose,ulrres,min_chunk_len,report=report)
    res_defined_as_SS = []
    for SS in SSs: res_defined_as_SS += SS
    
    for SS in additional_SS_def:
        # make sure residues do not duplicate
        valid = True
        for res in SS:
            if res in res_defined_as_SS:
                valid = False
                break
        if valid:
            SSs.append(SS)
            if report: print("additional SS %d-%d added to SSsegs."%(SS[0],SS[-1]))
        else:
            if report: print("additional SS %d-%d is not valid! skip."%(SS[0],SS[-1]))

    SSs.sort()
    if report: print( "SSsegs: ", SSs)
    
    cuts = []
    jumps = []
    movable = []
    SSid = {}
    SScen = [-1 for k in SSs]

    # chop by SS
    for i,SS1 in enumerate(SSs):
        SS1i = SS1[0]
        SS1f = SS1[-1]

        # select COM from res forming SSseg
        SSres = []
        for res in SS1:
            if SS3stat[res-1] != 'L': SSres.append(res)
            SSid[res] = i
        if len(SSres) < 3: SSres = range(SS1i,SS1f+1) #?
                
        SScen[i] = get_COMres(pose,SSres)
        print("SS %3d-%3d: cen %d"%(SS1[0],SS1[-1],SScen[i]))
        # assign cut
        if i != len(SSs)-1:
            SS2i = SSs[i+1][0]
            ulr_bw = None
            for ulr in ulrs:
                if ulr[0] > SS1f and ulr[-1] < SS2i:
                    ulr_bw = ulr
                    break
            cutpoint = int((SS1f+SS2i)/2) #SS2i-1
            cuts.append(cutpoint) #place cut at mid-point of loop
    
    # check which SS each chunk belongs to 
    #subid = {}
    #chunk_SSid = {}
    # simplified version
    SScovered = []
    for i,sub in enumerate(subs):
        #chunk_SSid[i] = []
        for j,chunk in enumerate(sub):
            #chunkcen = -1
            SScen_in_chunk = []
            for iSS,SS in enumerate(SSs):
                if SScen[iSS] in chunk:
                    #chunkcen = SScen[iSS]
                    SScen_in_chunk.append(SScen[iSS])
                    if iSS in SScovered:
                        sys.exit(SSs[iSS],' defined for multiple chunks! Check your chunk definition against SSs.\nSSs: ',SSs)
                    SScovered.append(iSS)
                    
            if len(SScen_in_chunk) == 0:
                cen = get_COMres(pose,chunk)
                print('Chunk %d-%d does not belong to any SS! Define a new one with center at %d'%(chunk[0],chunk[-1],cen))
                #sys.exit()
                # define a new one,..
                SScen_in_chunk = [cen]
                cuts.append(chunk[0])
                
            for k,cen in enumerate(SScen_in_chunk):
                if j == 0 and k == 0:
                    ancres = nres+1 #vroot
                    rootcen = cen
                    movable.append(True)
                else:
                    ancres = rootcen
                    movable.append(False)
                jumps.append((ancres,cen)) #place jump at central res
                print("Chunk/Sub %d/%d (%3d-%3d), Jump %3d-> %3d"%(i,j,chunk[0],chunk[-1],cen,ancres))
            
            #chunk_SSid[i].append(iSS)
            #for res in chunk:
                #subid[res] = (i,j)

    # fill undefined SSs
    for i,SS in enumerate(SSs):
        if i not in SScovered:
            movable.append(True)
            jumps.append((nres+1,SScen[i])) #place jump at central res
            print("Free SS (%3d-%3d), Jump %3d-> %3d"%(SS[0],SS[-1],SScen[i],nres+1))

    # add a cut before vroot
    cuts.append(nres)
           
    '''
    # assign jump: this is for each SS...
    for i,SS1 in enumerate(SSs):
        # see if cen belongs to any sub
        SSsub_id = (-1,-1)
        if SScen[i] in subid:
            isub,ichunk = subid[SScen[i]]
            if SSsub_id[1] == 0:
                ancres = nres+1 #vroot
            else:
                rootSS = chunk_SSid[isub][ichunk]
                ancres = 
                
        else: # when no sub definition exist for given SS
            ancres = nres+1
            
            jumps.append((ancres,SScen)) #place jump at central res
    '''
    

    if report:
        print( "Jumps: ", jumps)
        print( "Cuts: ", cuts)

    stat = tree_from_jumps_and_cuts(ft,nres+1,jumps,cuts,nres+1)

    if nres == pose.size():
        rosetta.core.pose.addVirtualResAsRoot(pose)
    rosetta.core.pose.symmetry.set_asymm_unit_fold_tree( pose, ft )
    rosetta.protocols.loops.add_cutpoint_variants( pose )
    return jumps, movable

def reset_fold_tree(pose, nres, saved_ft):
    pose.conformation().delete_residue_range_slow(nres+1, pose.size())
    pose.conformation().fold_tree(saved_ft)

    from pyrosetta.rosetta.core.chemical import CUTPOINT_LOWER, CUTPOINT_UPPER
    
    for i in range(1, pose.size()+1):
        if pose.residue(i).has_variant_type(CUTPOINT_LOWER):
            is_cut = False
            for ic in range(1, pose.fold_tree().num_cutpoint()+1):
                cutpoint = pose.fold_tree().cutpoint(ic)
                if i == cutpoint:
                    is_cut = True
                    break
            if not is_cut:
                pose_changed = True
                rosetta.core.pose.remove_variant_type_from_pose_residue(pose, CUTPOINT_LOWER, i)
                if pose.residue(i+1).has_variant_type(CUTPOINT_UPPER):
                    rosetta.core.pose.remove_variant_type_from_pose_residue(pose, CUTPOINT_UPPER, i+1)


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

