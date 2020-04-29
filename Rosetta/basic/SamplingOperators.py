from pyrosetta import *
import rosetta_utils
import random
import numpy as np
import copy
import time
from math import exp

class SamplingMaster:
    def __init__(self,opt,operators,probs,name):
        self.name = name
        self.operators = operators
        self.probs = probs
        self.minimizer = None
        self.minimize = False

    def apply(self,pose,move='random'):
        op_sel = None
        if move != 'random':
            for op in self.operators:
                if op.name == move:
                    op_sel = op
                    break
            if op_sel == None:
                print("Warning! such mover doesn't exist: %s; applying random"%move)
                
        if op_sel == None:
            op_sel = random.choices(self.operators,self.probs)[0]
            
        ans = op_sel.apply(pose)
        if ans != None: pose.assign(ans)
        
        if self.minimize:
            self.mimizer.apply(pose)
        return op_sel.name

    def select_mover(self):
        op_sel = random.choices(self.operators,self.probs)[0]
        return op_sel.name

    def show_ops(self):
        l = ' Sampler %s: Operators '%self.name
        for op in self.operators: l += ' %s'%op.name
        l += ', weights '
        for p in self.probs: l += ' %5.3f'%p
        print(l)

    def show(self):
        for op in self.operators:
            if hasattr(op,"show") and op.history_count != []:
                op.show()

class FragmentInserter:
    def __init__(self,opt,fragf,#alned_res,disallowed_res, ###unify to residue_weights!
                 residue_weights,
                 frag_ntop=25,name="",report=False):
        frag_IO = rosetta.core.fragment.FragmentIO( frag_ntop )
        nres = len(residue_weights)

        self.history_count = np.zeros(nres) #0-index
        self.fragset = frag_IO.read_data(fragf) #FragSet
        self.nmer = self.fragset.max_frag_length() #should be identical in length through the library

        self.residue_weights = residue_weights #[0.0 for i in range(nres)]
        self.last_inspos = nres-self.nmer+2
        self.name = name

        '''
        if opt.frag_insertion_mode == "weighted" and opt.resweights_fn != None:
            set_resweights_from_txt(opt.resweights_fn)

        elif opt.frag_insertion_mode == 'random':
            for res in range(1,self.last_inspos+1):
                nalned = 0.0
                for k in range(self.nmer):
                    if res+k in disallowed_res: #do not allow insertion through disallowed res
                        nalned = 0
                        break 
                    if res+k in alned_res: nalned += opt.aligned_frag_w
                    else: nalned += 1.0
                self.residue_weights[res-1] = nalned
        '''
                
        if report:
            print ("  [%s] insertion weight"%self.name,
                   "%3.1f "*len(self.residue_weights)%tuple(self.residue_weights))

    def apply(self,pose):
        insert_pos = random.choices(range(1,self.last_inspos+1),
                                    self.residue_weights[:self.last_inspos])[0]
        self.history_count[insert_pos-1] += 1

        #print( "INS %d"%insert_pos )
        mm = MoveMap()
        mm.set_bb(False)
        mm.set_bb(insert_pos)
        # decide where to put fragments
        #for k in range(self.nmer):
        #    mm.set_bb(insert_pos+k,True)

        #very stupid way but still works fast enough
        mover = rosetta.protocols.simple_moves.ClassicFragmentMover(self.fragset, mm)

        #fragsets = rosetta.utility_vector1_core_fragment_FragSet()
        #mover = rosetta.protocols.hybridization.WeightedFragmentTrialMover(fragsets,)
        mover.apply(pose)

    def show(self):
        l = ''
        for i,val in enumerate(self.history_count):
            if val > 0: l += ' %d:%d;'%(i+1,val)
        print(" [%s] insertion sites:"%self.name, l)

class FragmentMC:
    def __init__(self,fragf,residue_weights,
                 mciter=50,frag_ntop=25,
                 kT = 1.0,
                 npert=0,
                 w_chain_brk=1.0,
                 name=""):
        self.kT = kT
        
        frag_IO = rosetta.core.fragment.FragmentIO( frag_ntop )
        self.fragset = frag_IO.read_data(fragf) #FragSet
        self.nmer = self.fragset.max_frag_length() #should be identical in length through the library

        self.residue_weights = residue_weights #[0.0 for i in range(nres)]
        self.name = name
        self.scorer = create_score_function("score3.wts")
        self.scorer.set_weight(rosetta.core.scoring.linear_chainbreak, w_chain_brk)
        self.niter = mciter
        self.npert = npert
        
        nres = len(residue_weights)
        self.last_inspos = nres-self.nmer+2

    def apply(self,pose,report=False):
        if self.scorer == None:
            sys.exit("Scorer not defined for mover %s!"%self.name)

        time0 = time.time()
        # start with random insertion for perturbation...
        for it in range(self.npert):
            insert_pos = random.choices(range(1,self.last_inspos+1),
                                        self.residue_weights[:self.last_inspos])[0]
            mm = MoveMap()
            mm.set_bb(False)
            mm.set_bb(insert_pos)
            mover = rosetta.protocols.simple_moves.ClassicFragmentMover(self.fragset, mm)
            mover.apply(pose)
            
        pose_min = pose.clone()
        E0 = self.scorer.score(pose_min)
        Emin = E0
        Eprv = E0
        nacc = 0
        for it in range(self.niter):
            insert_pos = random.choices(range(1,self.last_inspos+1),
                                        self.residue_weights[:self.last_inspos])[0]
            mm = MoveMap()
            mm.set_bb(False)
            mm.set_bb(insert_pos)
            mover = rosetta.protocols.simple_moves.ClassicFragmentMover(self.fragset, mm)
            
            pose_work = pose.clone()
            mover.apply(pose_work)
            E = self.scorer.score(pose_work)
            
            if report:
                print("iter/res/Emin/E/Eprv: %3d %3d %8.3f %8.3f %8.3f"%(it,insert_pos,Emin,E,Eprv))
            if E < Eprv or np.exp(-(E-Eprv)/self.kT) > random.random():
                pose = pose_work
                nacc += 1
                Eprv = E
                
            if E < Emin:
                pose_min = pose
                Emin = E

        accratio = float(nacc)/self.niter
        dE = Emin-E0
        time1 = time.time()
        if report:
            print("Acc Ratio %.2f, Elapsed time for %d fragMC: %.1f sec"%(accratio,self.niter, time1-time0))
        return pose_min

class Chunk:
    def __init__(self, chunk_id=-1, threadable=False):
        self.id = chunk_id
        self.thread_s = list()
        self.coord = list()
        self.pos = None
        self.valid = True
        self.threadable = threadable
        self.history_count = []
        
    def read_chunk_info(self, line): ##supplemental
        x = line.split()
        if len(x) > 2:
            self.term_seq = x[2]
            self.align_pos = x[4]

    def read_thread_info(self, line, valid_ranges): ##supplemental
        x = line.split()
        thread_pos = [i for i in range(int(x[1]), int(x[2])+1)]
        thread_seq = ''
        if len(x) > 3:
            thread_seq = x[3]
        for reg in valid_ranges:
            if reg[0] <= thread_pos[0] and reg[-1] >= thread_pos[-1]:
                self.thread_s.append((thread_pos,thread_seq))
        
    def read_coord_info(self, line):
        x = line.split()
        crd = np.array(x[1:], dtype=np.float32)
        crd = crd.reshape((4,3))
        self.coord.append(crd)

    def thread_random(self):
        ithread = random.randrange(0,len(self.thread_s))
        self.pos = self.thread_s[ithread][0]

class ChunkReplacer:
    def __init__(self,opt):
        self.chunklib = []
        self.chunkprobs = []
        self.ulrs = opt.ulrs
        self.min_chunk_len = opt.min_chunk_len
        self.chunk_picked = None
        self.name = "ChunkReplacer"
        self.history_count = []

    def possible(self):
        return (len(self.chunklib) > 0)

    def read_chunk_from_pose(self,pose):
        ulrres = []
        for ulr in self.ulrs: ulrres += ulr

        # get SS seginfo from pose
        SSs,_ = rosetta_utils.pose2SSs(pose,ulrres,self.min_chunk_len)
        
        for i,SS in enumerate(SSs):
            chunk = Chunk(i+1)
            chunk.pos = SS
            for res in SS:
                rsd = pose.residue(res)
                crd = np.zeros((4,3))
                crd[0] = rsd.xyz("N")
                crd[1] = rsd.xyz("CA")
                crd[2] = rsd.xyz("C")
                crd[3] = rsd.xyz("O")
                chunk.coord.append(crd)
            self.chunklib.append(chunk)
            ## term_seq/align_pos/threads_s/
        self.chunkprobs = [1 for i in range(len(SSs))] # equal prob.

    def add_a_chunk(self,crds,threadres): #make directly
        chunk = Chunk(len(self.chunklib)+1)
        chunk.coord = crds
        chunk.pos = threadres
        self.chunklib.append(chunk)
        
    def read_chunk_from_extralib(self,txt):
        n_chunk = 0
        for line in open(txt):
            if line.startswith("CHUNK"):
                n_chunk += 1
                chunk = Chunk(n_chunk,threadable=True)
                chunk.read_chunk_info(line)
            elif line.startswith("THREAD"):
                chunk.read_thread_info(line,self.ulrs) #read in 
            elif line.startswith("COORD"):
                chunk.read_coord_info(line)
            elif line.startswith("END"):
                if chunk.thread_s != []:
                    self.chunklib.append(chunk)
        print( "Read in %d extra chunks from %s -- stored %d valid ones."%(n_chunk,txt,len(self.chunklib)))
        self.chunkprobs = [1 for i in self.chunklib] # equal prob.

    def idealize_geometry(self,pose,reslist,conntype):
        if conntype == "N":
            connres = reslist[-1] #pre-threadres
            pose.set_phi(connres+1,-150.0) #idealize first phi angle in thread
        elif conntype == "C":
            connres = reslist[0]-1 #last threadres
            pose.set_psi(connres, 150.0) #idealize last phi/omg angle in thread
            pose.set_omega(connres, 180.0)
        
        conf = pose.conformation()
        conf.insert_ideal_geometry_at_polymer_bond(connres)
    
        for res in reslist:
            pose.set_phi(res,-150.0)
            pose.set_psi(res, 150.0)
            pose.set_omega(res, 180.0)
            for iatm in range(4,pose.residue(res).natoms()+1):
                xyz = pose.residue(res).build_atom_ideal( iatm, pose.conformation() )
                pose.set_xyz( AtomID(iatm,res), xyz )


    def pick(self):
        chunk = random.choices(self.chunklib,self.chunkprobs)[0]
        if chunk.threadable: chunk.thread_random()
        self.chunk_picked = chunk
        return chunk
        
    def apply(self, pose):
        if self.chunk_picked == None:
            self.pick()
        chunk = self.chunk_picked
        
        for i_res, resNo in enumerate(chunk.pos):
            bb_crd = [rosetta.numeric.xyzVector_double_t(0.0) for i in range(4)] # N, CA, C, O
            crd = chunk.coord[i_res]
            for i in range(4):
                for j in range(3):
                    bb_crd[i][j] = crd[i,j]
                pose.set_xyz(AtomID(i+1,resNo), bb_crd[i])

            if pose.residue(resNo).natoms() > 4:
                for i_atm in range(5, pose.residue(resNo).natoms()+1):
                    xyz = pose.residue(resNo).build_atom_ideal(i_atm, pose.conformation())
                    pose.set_xyz(AtomID(i_atm, resNo), xyz)

        self.idealize_geometry(pose, [chunk.pos[0]-1], "N")
        self.idealize_geometry(pose, [chunk.pos[-1]+1], "C")
        self.chunk_picked = None #check as used
                
        '''
        #original
        if ulr[0] < chunk.pos[0]:
            idealize_geometry(pose, range(ulr[0], chunk.pos[0]), "N")
        if chunk.pos[-1] < ulr[-1]:
            idealize_geometry(pose, range(chunk.pos[-1]+1, ulr[-1]+1), "C")
        '''

class JumpSampler:
    def __init__(self,
                 anchors, #FTInfo,
                 maxtrans,
                 maxrot,
                 allowed_jumps=[],
                 SStypes=[],
                 name=""):

        # make a subset that are movable
        self.anchors = []
        if allowed_jumps == []:
            self.anchors = anchors
        else:
            self.anchors = [anchors[i] for i in allowed_jumps]

        # make sure len(SStypes) == len(anchors)
        if len(SStypes) != len(anchors):
            self.SStypes = ['H' for i in anchors] #Full scale for H, half for the rest
        else:
            self.SStypes = SStypes
                
        self.name = name
        self.maxtrans = maxtrans
        self.maxrot = maxrot
        self.history_count = []

    def apply(self,pose):
        njumps_movable = len(self.anchors)
        ancres = random.choices(self.anchors,[1.0 for k in range(njumps_movable)])[0]
        i = self.anchors.index(ancres)

        (maxrot,maxtrans) = (self.maxrot,self.maxtrans)
        if self.SStypes[i] != 'H':
            (maxrot,maxtrans) = (0.5*self.maxrot,0.5*self.maxtrans)

        jumpid = pose.fold_tree().get_jump_that_builds_residue( ancres )
        jump = pose.jump( jumpid )
        #print("JumpPert %d %d"%(jumpid,ancres))

        ## simpler way: is this good enough -- NEVER USE THIS
        ## direction: 1 or -1 -- meaning?
        #direction=1
        #jump.gaussian_move( direction, self.maxtrans, self.maxrot )

        ## direct control over T & R
        T = jump.get_translation()
        Q = rosetta_utils.R2quat( jump.get_rotation() )

        # random axis for now... may revisit later conditioning on SS type
        axis = np.random.rand(3)-0.5 #so that ranges within -0.5 to +0.5
        axis /= np.sqrt(np.dot(axis,axis))
        
        ang_in_rad = self.maxrot*random.random()*np.pi/180.0
        sa = np.sin(ang_in_rad)
       
        Qrot = [np.cos(ang_in_rad), sa*axis[0], sa*axis[1], sa*axis[2]]
        Qnew = rosetta_utils.Qmul( Q, Qrot )
        Rnew = rosetta_utils.quat2R( Qnew )

        Tnew = T
        for k in range(3):
            T[k] += self.maxtrans*(1.0-2.0*random.random())

        
        jump.set_translation( Tnew );
        jump.set_rotation( Rnew );
        
        pose.set_jump( jumpid, jump )

    def accept(self,pose,pose0):
        dev = np.zeros(pose.size())
        dd = 0.0
        for i in range(pose.size()):
            atmid = rosetta.core.id.AtomID(i+1,2) #CA position
            xyz0 = pose0.xyz(atmid)
            xyz1 = pose.xyz(atmid)
            dev = xyz0.distance(xyz1)
            dd += max(0, dev - self.dev_allowed[i])
        
        if dd < 0:
            return True
        return False

# deprecated
def perturb_pose_given_ft_setup(pose,FTInfo):
    # TODO!!!
    # residues to forget bb torsions
    # could we come up with a better logic through MinkTors??
    forget_bb = []
    
    # "Continuous ULR mode": Given max allowed deviation
    if FTInfo.CAdev != []:
        for i,jump in enumerate(FTInfo.jumps):
            if not jump.movable: continue
            
            sampler = JumpSampler(FTInfo,
                                  maxtrans=1.0,
                                  maxrot=1.0,
                                  allowed_jumps=[i])
            
            # try multiple times and pick one
            ntrial = 10
            pose0 = pose
            for i in range(ntrial):
                pose = pose0
                sampler.apply(pose)
                if sampler.accept(pose,pose0):
                    break

        # forget tors if CAdev predicted lower than threshold (Qres_lr <= 0.2)
        for i,val in enumerate(FTInfo.CAdev):
            if FTInfo.CAdev[i] > 99.9:
                forget_bb.append(i)

    # "Binary ULR mode": tors-to-forget comes from ulr definitions
    else:
        for ulr in FTInfo.ulrs:
            forget_bb += ulr

    # randomize torsion 
    for i in forget_bb:
        pose.set_phi(i+1,-150.0)
        pose.set_psi(i+1, 150.0)
        pose.set_omega(i+1, 180.0)
    
def get_residue_weights(opt,FTInfo,nres,nmer):
    disallowed = [] #FTInfo.cuts
    if not opt.verbose:
        print( "Fragment insertion disallowed: "+" %d"*len(disallowed)%tuple(disallowed))

    residue_weights = [0.0 for k in range(nres)]
    #print("ULRs:", FTInfo.ulrs)
    ulrres = []
    for ulr in FTInfo.ulrs: ulrres += list(ulr)

    if opt.frag_insertion_mode == "weighted":
        # 1. From user input
        if opt.resweights_fn != None:
            for l in open(opt.resweights_fn):
                words = l[:-1].split()
                resno = int(words[0])
                residue_weights[resno-1] = float(words[1])
            #return residue_weights
        
        # 2. Using MinkTors
        elif FTInfo.Qtors != [] and len(FTInfo.Qtors) == nres:
            print("Define residue weights from predicted torsion quality...")
            #log scale of CAdev, 1.0 at ULR, aligned_frag_w at core
            residue_weights = 1.0-FTInfo.Qtors
            residue_weights += -min(FTInfo.Qtors) + opt.aligned_frag_w
            #return residue_weights

        # 3rd. from CAdev estimation
        elif FTInfo.CAdev != []:
            print("Define residue weights from predicted CA deviation...")
            residue_weights = np.log(FTInfo.CAdev/100.0+1.0) + opt.aligned_frag_w #log scale of CAdev, 1.0 at ULR, aligned_frag_w at core
            #return residue_weights
        
    else:
        # Otherwise binary
        print("Weighted Frag Insertion failed: cannot detect resweights_fn nor AutoCAdev from .npz. Do random instead.")

        for res in range(1,nres-nmer+2):
            nalned = 0.0
            for k in range(nmer):
                if res+k in disallowed: #do not allow insertion through disallowed res
                    nalned = 0
                    break 
                if res+k not in ulrres: nalned += opt.aligned_frag_w
                else: nalned += 1.0
            residue_weights[res-1] = nalned
            
    l = 'RESIDUE WEIGHTS: '
    for k in range(nres):
        if k%20 == 0: l += '\n%3d- :'%(k+1)
        l += ' %4.2f'%(residue_weights[k])
    print(l)
    return residue_weights

def get_samplers(pose,opt,FTInfo=None):
    opt_cons = copy.copy(opt)
    opt_cons.aligned_frag_w = 0.0
    
    main_units = []
    main_w = []
    pert_units = []
    pert_w = []
    refine_units = []
    refine_w = []

    nres = pose.size()-1
    
    residue_weights_big = get_residue_weights(opt,FTInfo,nres,9)
    residue_weights_small = get_residue_weights(opt,FTInfo,nres,3)

    # Get pose SS info
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    SS3 = [a for a in dssp.get_dssp_reduced_IG_as_L_secstruct()] # as a string
    
    # get list of jumps to sample
    jumps_to_sample = [i for i,jump in enumerate(FTInfo.jumps) if jump.movable]
    print("Jumps to sample",  jumps_to_sample)

    mover_weights = opt.mover_weights
    if mover_weights == []: #auto
        frag_w = np.mean(residue_weights_big)/9.0 #0~1
        jump_w = np.sum([FTInfo.jumps[i].nres for i in jumps_to_sample])/FTInfo.nres
        mover_weights = np.array([frag_w,jump_w,0.0])
        mover_weights /= np.sum(mover_weights)
        print("- Mover weight undefined; Set auto weight: %5.3f/%5.3f/%5.3f"%tuple(mover_weights))
    
    '''
    residue_weights_close = np.zeros(nres)+0.1
    for cut in FTInfo.cuts:
        for k in range(-9,10):
            if cut+k < 1 or cut+k >= nres: continue
            residue_weights_close
    '''
    
    ## Samplers
    multiFragInsertor_p = FragmentMC(opt.fragbig,
                                     residue_weights_big,
                                     mciter=25, #mciter x niter x nsample = 20x10x50 = 10000
                                     kT=2.0,npert=0,w_chain_brk=1.0,
                                    name="FragMCBig")
    multiFragInsertor_small = FragmentMC(opt.fragsmall,
                                         residue_weights_small,
                                         mciter=25, #mciter x niter x nsample = 20x10x50 = 10000
                                         kT=2.0,npert=0,w_chain_brk=1.0,
                                         name="FragMCSmall")
    
    # MultifragMCmover for closure
    multiFragInsertor_c = FragmentMC(opt.fragbig,
                                    residue_weights_big,
                                    mciter=20, #mciter x niter x nsample = 20x10x50 = 10000
                                    name="FragMCBig")
    closer = SamplingMaster(opt,[multiFragInsertor_c],[1.0],"Closer")

    # Fragment insertion
    if mover_weights[0] > 1.0e-6:
        w = mover_weights[0]

        # Mutation operator for initial perturbation
        mutOperator0 = FragmentInserter(opt_cons,opt.fragbig,residue_weights_big,
                                        name="FragBigULR")
    
        # Regular Mutation operator with big-frag insertion
        mutOperator1 = FragmentInserter(opt,opt.fragbig,residue_weights_big,
                                        name ="FragBig")
        # Regular mutation operator with small-frag insertion
        mutOperator2 = FragmentInserter(opt,opt.fragsmall,residue_weights_small,
                                        name="FragSmall")

        # Chunk from template poses -- unimplemented yet
        #chunkOperator = ChunkReplacer(opt)
        #chunkOperator.read_chunk_from_pose(pose)

        pert_units += [mutOperator0]
        pert_w += [w]

        if opt.fragopt == "mc":
            main_units += [multiFragInsertor_p]
            main_w += [1.0]
            refine_units += [multiFragInsertor_small]
            refine_w += [w]
        else:
            main_units += [mutOperator1, mutOperator2] #, chunkOperator]
            main_w += [w*opt.p_mut_big, w*opt.p_mut_small] # == 1,0 by default #, w*opt.p_chunk]
            refine_units += [mutOperator2]
            refine_w += [w]
        
        print("Make Fragment sampler with weight %.3f..."%w)

    # Jump mover
    # Segment searcher also part of here, stub defs passed through FTInfo
    if mover_weights[1] > 1.0e-6:
        w = mover_weights[1]
        jumpcens = [jump.cen for jump in FTInfo.jumps]
        allowed_jumps = [i for i,jump in enumerate(FTInfo.jumps) if jump.movable]
        SStypes = [SS3[jump.cen-1] for jump in FTInfo.jumps]
        perturber = JumpSampler(jumpcens,
                                2.0,15.0,
                                allowed_jumps=allowed_jumps,
                                SStypes=SStypes,
                                name="JumpBig")
        refiner   = JumpSampler(jumpcens,
                                1.0,5.0,
                                allowed_jumps=allowed_jumps,
                                SStypes=SStypes,
                                name="JumpSmall")
        pert_units += [perturber]
        pert_w += [w]
        main_units += [perturber]
        main_w += [w]
        refine_units += [refiner]
        refine_w += [w]
        print("Make     Jump sampler with weight %.3f..."%w)

    #print( [unit.name for unit in main_units] )
    #print( main_w )
    perturber = SamplingMaster(opt_cons,pert_units,pert_w,"perturber")
    mainmover = SamplingMaster(opt,main_units,main_w,"main") #probability
    refiner = SamplingMaster(opt,refine_units,refine_w,"refiner")

    # MinMover
    sf = create_score_function("score4_smooth")
    mmap = MoveMap()
    mmap.set_bb(False)
    mmap.set_jump(False)
    #minimize only relevant DOF to make minimizer faster
    for i,p in enumerate(residue_weights_small):
        if p > 1.0: mmap.set_bb(i+1,True)
    for jumpno in jumps_to_sample:
        mmap.set_jump(jumpno+1,True)
        
    minimizer = rosetta.protocols.minimization_packing.MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True) 
    minimizer.max_iter(5)
    mainmover.minimizer = minimizer
    refiner.minimizer = minimizer
    
    return perturber, mainmover, refiner, closer
