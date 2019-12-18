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
    def __init__(self,opt,anchors,movable,
                 maxtrans,maxrot,
                 name=""):
        self.name = name

        self.anchors = []
        for i,res in enumerate(anchors):
            if movable[i]: self.anchors.append(res)
            
        self.maxtrans = maxtrans
        self.maxrot = maxrot

    def apply(self,pose):
        #print("anchors", self.anchors )
        njumps_movable = len(self.anchors)
        ancres = random.choices(self.anchors,[1.0 for k in range(njumps_movable)])[0]

        jumpid = pose.fold_tree().get_jump_that_builds_residue( ancres )
        jump = pose.jump( jumpid )

        ## direction: 1 or -1 -- meaning?
        direction = 1
        #simpler way: is this good enough
        jump.gaussian_move( direction, self.maxtrans, self.maxrot )

        '''
        # explicit calling
        # rotation
        R = rosetta.numeric.xyzMatrix()
        jump.set_rotation( R )

        # translation
        tv = maxtrans*random.random()
        jump.random_trans( tv )
        '''

        pose.set_jump( jumpid, jump )

