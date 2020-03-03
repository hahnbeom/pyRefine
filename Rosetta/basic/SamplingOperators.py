from pyrosetta import *
import rosetta_utils
import random
import numpy as np

class SamplingMaster:
    def __init__(self,opt,operators,probs,name):
        self.name = name
        self.operators = operators
        self.probs = probs

    def apply(self,pose):
        op_sel = random.choices(self.operators,self.probs)[0]
        op_sel.apply(pose)
        return op_sel.name

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
            print ("%s, insertion weight"%self.name,
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
        mover.apply(pose)

    def show(self):
        l = ''
        for i,val in enumerate(self.history_count):
            if val > 0: l += ' %d:%d;'%(i+1,val)
        print("Insertion sites:", l)
        
class Chunk:
    def __init__(self, chunk_id=-1, threadable=False):
        self.id = chunk_id
        self.thread_s = list()
        self.coord = list()
        self.pos = None
        self.valid = True
        self.threadable = threadable
        
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
                 name=""):

        # make a subset that are movable
        self.anchors = []
        if allowed_jumps == []:
        #    for i,jump in enumerate(FTInfo.jumps):
        #        if jump.movable: self.anchors.append(anc)
            self.anchors = anchors
        else:
            #for i in allowed_jumps:
            #    jump = FTInfo.jumps[i]
            #    if jump.movable: self.anchors.append(jump.anchor)
            self.anchors = [anchors[i] for i in allowed_jumps]
                
        self.name = name
        self.maxtrans = maxtrans
        self.maxrot = maxrot

    def apply(self,pose):
        njumps_movable = len(self.anchors)
        ancres = random.choices(self.anchors,[1.0 for k in range(njumps_movable)])[0]

        jumpid = pose.fold_tree().get_jump_that_builds_residue( ancres )
        jump = pose.jump( jumpid )

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
    
    
    
