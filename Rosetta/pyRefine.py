import sys,os
import copy,random,math
import pyrosetta as PR #call explicitly
import SamplingOperators as SO
import rosetta_utils
import estogram2cst

# initialize pyRosetta
#should be initialized in main
initcmd = '-hb_cen_soft -overwrite -mute all'
PR.init(initcmd)

def arg_parser(argv):
    import argparse
    
    opt = argparse.ArgumentParser\
          (description='''FragAssembler: Assemble fragments for defined ULRs''')
    ## Input
    opt.add_argument('-s', dest='pdb_fn', metavar='PDB_FN', required=True, \
                     help='Input pdb file')
    opt.add_argument('-frag_fn_big', dest='fragbig', metavar='FRAGBIG', required=True,\
                     help='Fragment library file, big')
    opt.add_argument('-frag_fn_small', dest='fragsmall', metavar='FRAGSMALL', required=True,\
                     help='Fragment library file, small')
    opt.add_argument('-ss_fn', dest='ss_fn', metavar="SS_FN", default=None,
                     help='SS prediction file')
    opt.add_argument('-chunk_fn', dest='chunk_fn', metavar='CHUNK_FN', default=None,
                     help="TERM chunk search result file")
    opt.add_argument('-native', dest='native', default=None, help='Input ref file')
    opt.add_argument('-ulr', dest='ulr_s', metavar='ULRs', nargs='+', default=[], \
                     help='ULRs should be sampled. (e.g. 5-10 16-20). If not provided, whole structure will be sampled')

    ## Output
    opt.add_argument('-prefix', dest='prefix', default="S", \
                     help='Prefix of output pdb file')
    opt.add_argument('-nstruct', dest='nstruct', type=int, default=1,
                     help='number of structures to sample serially')
    opt.add_argument('-cen_only', dest='cen_only', default=False, action="store_true",
                     help='Pass all-atom stage')
    opt.add_argument('-write_pose', dest='write_pose', default=True, action='store_true', \
                     help="Write output structures [False]")
    opt.add_argument('-dump_trj_every', dest='dump_trj_every', type=int, default=1e10, \
                     help="Write trajectory every n steps [1e10]")
    opt.add_argument('-outsilent', dest='outsilent', default=None, 
                     help="Write output poses at given silent file")
    opt.add_argument('-debug', dest='debug', default=False, action='store_true', \
                     help='Debug mode [turn off]')
    opt.add_argument('-verbose', dest='verbose', default=False, action='store_true', \
                     help='Verbose mode [turn off]')
    opt.add_argument('-mute', dest='mute', default=False, action='store_true', \
                     help='Mute all [turn off]')

    ## Scoring options
    opt.add_argument('-score', dest='scoretype', metavar="SCORE", default="score3",\
                     help='centroid Score type')
    opt.add_argument('-cen_cst', dest='cen_cst', default=None, help='centroid restraint file')
    opt.add_argument('-fa_cst', dest='fa_cst', default=None, help='fullatom restraint file')
    opt.add_argument('-cen_cst_w', dest='cen_cst_w', type=float, default=1.0, help='')
    opt.add_argument('-fa_cst_w', dest='fa_cst_w', type=float, default=1.0, help='')
    opt.add_argument('-refcorrection_method', dest='refcorrection_method', default="none", help='')
    
    ## Sampling options
    #### relative weights
    opt.add_argument('-mover_weights', dest='mover_weights', nargs='+', type=float, default=[1.0,0.0,0.0], #fraginsert only
                     help="weights on movers [frag-insert/jump-opt/motif-insert]")
    
    #### Jump-opt
    #opt.add_argument('-chunk_cut', dest='chunk_cut', nargs='+', type=int, default=[],
    #                 help="location of chunk cut positions")
    opt.add_argument('-sub_def', dest='sub_def', nargs='+', default=[],
                     help="definition of subs: argument same as -ulr")
    
    #### FragInsert
    opt.add_argument('-min_chunk_len', dest='min_chunk_len', nargs='+', type=int, default=[3,5,1000],
                     help="minimum length of SS defined as chunk [E/H/C]")

    opt.add_argument('-variable_cutpoint', dest='variable_cutpoint', default=False,
                     help="Randomly pick cutpoint instead of loop-center")
    opt.add_argument('-variable_nachor', dest='variable_anchor', default=False,
                     help="Randomly pick chunk anchor instead of chunk-COM")

    opt.add_argument('-sch_fn', dest='sch_fn', type=str, default='', help='Defines MC schedule')
    
    opt.add_argument('-recover_min_every', dest='recover_min_every', type=int, default=100, \
                     help='Recover min energy pose every this iteration during MC')

    opt.add_argument('-frag_insertion_mode', dest='frag_insertion_mode', default="random", \
                     help='Fragment insertion weighting scheme')
    opt.add_argument('-aligned_frag_w', dest='aligned_frag_w', type=float, default=0.0,
                     help='weight on aligned region (i.e. non-ulr) for fragment insertion')

    opt.add_argument('-p_mut_big',   dest='p_mut_big',   default=  1.0, help="prob. frag_big insertion")
    opt.add_argument('-p_mut_small', dest='p_mut_small', default=  0.0, help="prob. frag_small insertion")
    opt.add_argument('-p_chunk',     dest='p_chunk',     default=  0.0, help="prob. chunk insertion")
    opt.add_argument('-kT',          dest='kT0',          type=float, default=  2.0,
                     help="MC temperature factor")
    opt.add_argument('-batch_per_relax', dest='batch_per_relax', type=int, default=1,
                     help='number of structures to sample in centroid per each relax run')

    # Fragment library option
    opt.add_argument('-resweight_fn', dest='resweight_fn', default=None,
                     help='per-res Fragment insertion weight input')
    #opt.add_argument('-fragbig_ntop', dest='fragbig_ntop', default=25,)
    #opt.add_argument('-fragsmall_ntop', dest='fragsmall_ntop', default=25,)
    
    #
    if len(argv) < 1:
        opt.print_help()
        sys.exit(1)

    # support for Rosetta style flags file
    # should support multiple flags (e.g. @flag1 @flag2 -s init.pdb ...)
    for arg in copy.copy(argv):
        if arg[0] == '@':
            if not os.path.exists(arg[1:]):
                sys.exit('Flags file "%s" does not exist!'%arg[1:])
            for l in open(arg[1:]):
                if not l.startswith('-'): continue
                argv += l[:-1].strip().split()
            argv.remove(arg)

    params = opt.parse_args(argv)
    
    # post-process
    params.subs = []
    for i,substr in enumerate(params.sub_def):
        chunks = substr.split(',')
        sub = []
        for chunk in chunks:
            begin = int(chunk.split('-')[0])
            end   = int(chunk.split('-')[-1])
            sub.append(list(range(begin,end+1)))
        params.subs.append( sub )
    params.ulrs = []
    for i,ulrstr in enumerate(params.ulr_s):
        params.ulrs.append( list(range( int(ulrstr.split('-')[0]), int(ulrstr.split('-')[-1])+1 )) )
            
    return params


def get_residue_weights_from_opt(opt,disallowed,nres,nmer):
    if not opt.verbose: print( "Fragment insertion disallowed: "+" %d"*len(disallowed)%tuple(disallowed))

    residue_weights = [0.0 for k in range(nres)]
    ulrres = []
    for ulr in opt.ulrs: ulrres += ulr

    if opt.frag_insertion_mode == "weighted" and opt.resweights_fn != None:
        # should replace by lddt prediction...
        for l in open(opt.resweights_fn):
            words = l[:-1].split()
            resno = int(words[0])
            residue_weights[resno-1] = float(words[1])

    elif opt.frag_insertion_mode == 'random': #classic way!
        for res in range(nres-nmer+2):
            nalned = 0.0
            for k in range(nmer):
                if res+k in disallowed: #do not allow insertion through disallowed res
                    nalned = 0
                    break 
                if res+k not in ulrres: nalned += opt.aligned_frag_w
                else: nalned += 1.0
            residue_weights[res-1] = nalned
    return residue_weights

def get_samplers(pose,opt,jump_anchors,
                 anchors_movable=None):
    opt_cons = copy.copy(opt)
    opt_cons.aligned_frag_w = 0.0

    # make sure the length of jump_anchors == anchors_movable
    if anchors_movable != None:
        if len(jump_anchors) != len(anchors_movable):
            sys.exit("Inconsistent anchor- & movable- definitions!!")
    else:
        anchors_movable = [True for anc in jump_anchors] # move all if not specified
    
    main_units = []
    main_w = []
    pert_units = []
    pert_w = []
    refine_units = []
    refine_w = []

    nres = pose.size()-1
    
    if opt.mover_weights[0] > 1.0e-6:
        w = opt.mover_weights[0]

        residue_weights_big = get_residue_weights_from_opt(opt,jump_anchors,nres,9) ## hard-coded!!
        residue_weights_small = get_residue_weights_from_opt(opt,jump_anchors,nres,3)
        
        # Mutation operator for initial perturbation
        mutOperator0 = SO.FragmentInserter(opt_cons,opt.fragbig,residue_weights_big,
                                           name="FragBigULR")
    
        # Regular Mutation operator with big-frag insertion
        mutOperator1 = SO.FragmentInserter(opt,opt.fragbig,residue_weights_big,
                                           name ="FragBig")
        # Regular mutation operator with small-frag insertion
        mutOperator2 = SO.FragmentInserter(opt,opt.fragsmall,residue_weights_small,
                                           name="FragSmall")

        # Chunk from template poses -- unimplemented yet
        chunkOperator = SO.ChunkReplacer(opt)
        chunkOperator.read_chunk_from_pose(pose)

        pert_units += [mutOperator0]
        pert_w += [w]
        main_units += [mutOperator1, mutOperator2, chunkOperator]
        main_w += [w*opt.p_mut_big, w*opt.p_mut_small, w*opt.p_chunk]
        refine_units += [mutOperator2]
        refine_w += [w]
        print("Make Fragment sampler with weight %.3f..."%w)

    if opt.mover_weights[1] > 1.0e-6:
        w = opt.mover_weights[1]
        perturber = SO.JumpSampler(opt,jump_anchors,anchors_movable,
                                   1.0,5.0,name="JumpBig")
        refiner   = SO.JumpSampler(opt,jump_anchors,anchors_movable,
                                   0.5,2.0,name="JumpSmall")
        pert_units += [perturber]
        pert_w += [w]
        main_units += [perturber]
        main_w += [w]
        refine_units += [refiner]
        refine_w += [w]
        print("Make     Jump sampler with weight %.3f..."%w)

    # ChunkReplacer TODO

    print( [unit.name for unit in main_units] )
    print( main_w )
    perturber = SO.SamplingMaster(opt_cons,pert_units,pert_w,"perturber")
    mainmover = SO.SamplingMaster(opt,main_units,main_w,"main") #probability
    refiner = SO.SamplingMaster(opt,refine_units,refine_w,"refiner")

    return perturber, mainmover, refiner

# One step of MC in FT hybridize
def MC(pose,scorer,sampler,opt,tot_it,
       recover_min=True,recover_min_every=1e6,
       stageno=0):
    
    verbose = opt.verbose
    Eprv = scorer.score(pose)
    Emin = Eprv
    nacc = 0
    pose_min = pose
    accratio = 0.0
    for it in range(tot_it):
        if it%100 == 0:
            kT = scorer.autotemp(it, tot_it, accratio)
        if it%recover_min_every == 0:
            pose = pose_min

        pose_work = pose.clone()
        tag = sampler.apply(pose_work)
            
        E = scorer.score(pose_work)
        accepted = False
        if E < Eprv or math.exp(-(E-Eprv)/kT) > random.random():
            accepted = True

        if accepted:
            pose = pose_work
            nacc += 1
        if E < Emin:
            pose_min = pose
            Emin = E

        accratio = float(nacc)/(it+1)
        if verbose and (it%10 == 0 or it == tot_it-1):
            print( "it/Type/kT/score: %4d %-12s %5.1f %9.3f %9.3f %9.3f %5.3f"%(it,tag,kT,E,Eprv,Emin,accratio) )
        if accepted: Eprv = E
        if it%opt.dump_trj_every == opt.dump_trj_every-1:
            pose.dump_pdb("trj%02d.%04d.pdb"%(stageno,it+1))
        
    if recover_min: pose = pose_min
    return pose, accratio

# Reproduced FT hybridize in Rosetta
def FoldTreeSample(pose,opt,refpose=None,aln=None,mute=False):
    nres = pose.size()
    ulrs = opt.ulrs
    print( "ULR:", ulrs )

    # 0. Setup initial chunk insertion from extralib
    chunkOperator_extra = None
    chunks_pre_insert = []
    if opt.chunk_fn != None:
        chunkOperator_extra = SO.ChunkReplacer(opt)
        chunkOperator_extra.read_chunk_from_extralib(opt.chunk_fn)
        if chunkOperator_extra.possible():
            chunk_pre_insert = chunkOperator_extra.pick()
            for i,ulr in enumerate(ulrs):
                if chunk_pre_insert.pos[0] in ulr and chunk_pre_insert.pos[-1] in ulr:
                    for res in chunk_pre_insert.pos:
                        ulrs[i].remove(res)
                break
            if opt.verbose: print( "Pre-insert at ", chunk_pre_insert.pos )
            chunks_pre_insert = [chunk_pre_insert.pos]
        else:
            print( "None of the extra chunks read is compatible with ULR definition. Skip pre-insertion stage.")

    ## 1. FoldTree setup
    jumps,movable = rosetta_utils.setup_fold_tree(pose,ulrs,opt,
                                                  additional_SS_def=chunks_pre_insert)
    # 1a. Then apply insertion
    if chunks_pre_insert != []:
        chunkOperator_extra.apply(pose) #start with a chunk inserted
        
    jump_anchors = [jump[1] for jump in jumps] #anchor_res at chunk
    if opt.debug: pose.dump_pdb("ftsetup.pdb")

    ## 2. Sampling operators
    # originally fragment insertion prohibitted at jump_anchors
    # -- should revisit?? (or, instead on cuts)
    perturber, inserter, refiner = get_samplers(pose,opt,jump_anchors,
                                                anchors_movable=movable)
       
    # 3. Score setup: term weights adjusted by scheduler
    scorer = Scorer(opt) 
    
    if not mute: print( "Starting score: %8.3f"%scorer.score(pose))
    # 4. pre-MC:
    ## slightly randomize at the beginning; make extended at ULR
    for ulr in ulrs:
        for res in ulr:
            pose.set_phi(res,-150.0)
            pose.set_psi(res, 150.0)
            pose.set_omega(res, 180.0)
    if opt.debug: pose.dump_pdb("stage0.pdb")
    
    # 5. MC
    scheduler = Scheduler(opt)
    #if opt.sch_fn != '':
    #    scheduler = read_scheduler_from_file(opt.sch_fn)

    for it in range(scheduler.nstages()):
        scheduler.adjust_scorer(it,scorer)
        niter = scheduler.get_niter(it)
        if not mute: print("Running MC stage%d with n_iter %d..."%(it+1,niter))

        if it == 0:
            sampler = perturber
        elif it == scheduler.nstages()-1:
            sampler = refiner
        else:
            sampler = inserter

        score0 = scorer.score(pose)
        pose,accratio = MC(pose,scorer,sampler,opt,niter,
                           stageno=it)
        
        chain_brk = scorer.get_term(pose, PR.rosetta.core.scoring.linear_chainbreak)
        
        l = "Stage%d (Acceptance %5.3f): %8.3f -> %8.3f, chainbrk: %8.3f"%(it+1,accratio,score0,scorer.score(pose),chain_brk)
        if scorer.Edssp < 0:
            l += ', DSSP %8.3f'%scorer.Edssp

        if refpose != None:
            gdtmm = PR.rosetta.protocols.hybridization.get_gdtmm(refpose,pose,aln)
            l += ', GDTMM %5.3f'%gdtmm
        
        if not mute: print( l )
        if opt.debug: pose.dump_pdb("stage%d.pdb"%(it+1))
    
    return pose, scorer.score(pose)

# writing PDB 
def report_pose(pose,tag,extra_score,outsilent,mute=False):
    # dump to silent
    if outsilent != None:
        silentOptions = PR.rosetta.core.io.silent.SilentFileOptions()
        sfd = PR.rosetta.core.io.silent.SilentFileData(silentOptions)
        
        if not mute: print( "Reporting pose to silent %s..."%outsilent )
        ss = PR.rosetta.core.io.silent.BinarySilentStruct(silentOptions,pose)
        ss.set_decoy_tag(tag)
        for (key,val) in extra_score:
            ss.add_energy(key,val) #add additional info
        sfd.write_silent_struct(ss,outsilent)
    # dump to individual pdb
    else:
        if not mute: print( "Reporting pose to pdb %s..."%tag)
        pose.dump_pdb(tag)

class Scorer:
    def __init__(self,opt):
        sfxn = PR.create_score_function("score3")
        sfxn.set_weight(PR.rosetta.core.scoring.cen_hb, 5.0)
        self.Edssp = 1e6
        self.wdssp = 5.0
        self.kT_mulfactor = 1.0
        self.sfxn = sfxn
        self.ss8 = []
        if opt.ss_fn != None:
            self.ss8 = np.load(opt.ss_fn)
            # corrections to termini
            self.ss8[0, :] = 1.0 / 8.0
            self.ss8[-1, :] = 1.0 / 8.0
            self.ss3 = np.zeros((len(self.ss8),3))
            #BEGHIST_: H 2,3,4; E 0,1; L 5,6,7
            self.ss3[:,0] = np.sum([self.ss8[:,2],self.ss8[:,3],self.ss8[:,4]],axis=0)
            self.ss3[:,1] = np.sum([self.ss8[:,0],self.ss8[:,1]],axis=0)
            self.ss3[:,2] = np.sum([self.ss8[:,5],self.ss8[:,6],self.ss8[:,6]],axis=0)
            
    def calc_dssp_agreement_score(self,pose,res_s):
        dssp = PR.rosetta.core.scoring.dssp.Dssp(pose)
        ss8_type = np.array(list("BEGHIST "), dtype='|S1').view(np.uint8)
        ss3_type = np.array(list("HEL"), dtype='|S1').view(np.uint8)
        dssp8 = np.array(list(dssp.get_dssp_unreduced_secstruct()), dtype='|S1').view(np.uint8)
        dssp3 = np.array(list(dssp.get_dssp_secstruct()), dtype='|S1').view(np.uint8)
        
        for i in range(ss8_type.shape[0]):
            dssp8[dssp8 == ss8_type[i]] = i
            dssp8[dssp8 > 7] = 7 # coil

        for i in range(ss3_type.shape[0]):
            dssp3[dssp3 == ss3_type[i]] = i

        E = 0.0
        for res in res_s:
            #E -= self.ss8[res-1,dssp8[res-1]]
            E -= self.ss3[res-1,dssp3[res-1]]
        return E*self.wdssp

    def reset_wts(self,scoretype,wts):
        self.sfxn.set_weight(scoretype, wts)
    
    def get_term(self,pose,scoretype):
        return pose.energies().total_energies()[scoretype]

    def reset_kT(self,val):
        self.kT_mulfactor = 1.0
        self.kT0 = val

    def autotemp(self,it,tot_it,accratio):
        '''
        f_it = float(it)/tot_it
        if f_it < 0.25:
            pass
        else:
            if accratio > 0.5:
                self.kT_mulfactor *= 0.5
            elif accratio < 0.1:
                self.kT_mulfactor *= 2.0
        ''' #constant
        return self.kT0*self.kT_mulfactor

    def score(self,pose):
        self.Edssp = 0.0
        self.E = self.sfxn.score(pose)
        if self.ss8 != []:
            #for res in range(pose.size()):
            #TODO: score only ulr? 
            self.Edssp += self.calc_dssp_agreement_score(pose,range(1,pose.size()))
        return self.E + self.Edssp

class Scheduler:
    def __init__(self,opt):
        #self.load_dflt_sch()
        self.load_dflt_sch()
        if opt.debug:
            self.load_debug_sch()
        self.kT   = [opt.kT0*kT for kT in self.kT]
        self.wcst = [w*opt.cen_cst_w for w in self.wcst]

    def load_dflt_sch(self):
        # define a default schedule
        #originally in hybrid
        # stage1: score0, vdw0.1 brk0  -> 1000
        # stage2: score1, vdw1   brk0.1 cst0.1 -> 1000
        # stage3: {score2<->score5}, vdw1  brk0.25 cst0.25 -> 10*1000
        # stage4: score3, full scale -> 3*200
        self.niter = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        self.kT    = [2.5,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.wvdw  = [0.1,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.wbrk  = [0.0,  0.1, 0.1, 0.3, 0.2, 0.7, 0.5, 1.0, 0.8, 1.0, 1.0]
        self.wcst  = [0.0,  0.1,0.25,0.25,0.25,0.25,0.25,0.25, 0.25,0.25,1.0]

    def load_debug_sch(self):
        self.niter = [100 , 500, 500,  500]
        #self.kT    = [2.5,  1.0, 1.0, 1.0]
        self.kT    = [2.5,  2.5, 2.5, 2.5]
        self.wvdw  = [0.1,  1.0, 1.0, 1.0]
        self.wbrk  = [0.0,  0.1, 0.5, 1.0]
        self.wcst  = [0.0,  0.1,0.25, 1.0]

    def nstages(self):
        return len(self.niter)
        
    def adjust_scorer(self,it,scorer):
        scorer.reset_kT( self.kT[it] )
        scorer.reset_wts( PR.rosetta.core.scoring.vdw, self.wvdw[it] )
        scorer.reset_wts( PR.rosetta.core.scoring.linear_chainbreak,    self.wbrk[it] )
        scorer.reset_wts( PR.rosetta.core.scoring.atom_pair_constraint, self.wcst[it] )

    def get_niter(self,it):
        return self.niter[it]

class Runner:
    def __init__(self,params=None):
        if params != None:
            self.opt = params
        else:
            self.opt = arg_parser(sys.argv[1:])
            
    def apply(self,pose0=None):
        # pose0 can be provided through argument or optparser
        if pose0 == None and self.opt.pdb_fn != None:
            pose0 = PR.pose_from_file(self.opt.pdb_fn)

        if pose0 == None:
            sys.exit("No starting pose defined!")
    
        PR.SwitchResidueTypeSetMover("centroid").apply(pose0)

        # Get native pose & seqaln to it
        refpose = None
        aln = None
        if self.opt.native != None:
            refpose = PR.pose_from_file(self.opt.native)
            natseq = PR.rosetta.core.sequence.Sequence( refpose.sequence(),"native",1 ) 
            seq    = PR.rosetta.core.sequence.Sequence( pose0.sequence(),"model",1 ) 
            aln = PR.rosetta.core.sequence.align_naive(seq,natseq)
            
        # store original FT
        ft = pose0.conformation().fold_tree().clone()

        # Cen constraints
        if self.opt.cen_cst.endswith('npz'): # from DL
            #temporary
            self.opt.Pcore = [0.6,0.7,0.8]
            self.opt.Pspline_on_fa=0.0 #
            self.opt.Pcontact_cut =0.9
            self.opt.hardcsttype="bounded",
            self.refcorrection_method="statQ"
            
            estogram2cst.apply_on_pose( pose0,npz=self.opt.cen_cst,
                                        opt=self.opt )
            
        elif self.opt.cen_cst != None:
            constraints = PR.rosetta.protocols.constraint_movers.ConstraintSetMover()
            constraints.constraint_file(opt.cen_cst)
            constraints.apply(pose0)

        nsample_cen = self.opt.nstruct*self.opt.batch_per_relax

        # Repeat Centroid-MC nsample_cen times
        poses_store = []
        for modelno in range(nsample_cen):
            pose = pose0.clone()
            if not self.opt.mute: print("Generating %d/%d structure..."%(modelno+1,nsample_cen))
            # Coarse-grained sampling stage
            pose,score = FoldTreeSample(pose,self.opt,refpose,aln,self.opt.mute) 

            # recover original fully-connected FT
            rosetta_utils.reset_fold_tree(pose,pose.size()-1,ft)
            pose.remove_constraints()
            poses_store.append((score,pose))

            if not self.opt.mute and refpose != None:
                gdtmm = PR.rosetta.protocols.hybridization.get_gdtmm(refpose,pose,aln)
                print("Centroid score: %8.3f, GDTMM %5.3f"%(score,gdtmm))
            else:
                print("Centroid score: %8.3f"%score)

        poses_store.sort()
        batch = [(score,pose) for (score,pose) in poses_store[:self.opt.nstruct]]

        if not self.opt.mute:
            print("Relaxing %d selected structures with cen_score range %.1f~%.1f."\
                  %(self.opt.nstruct,batch[0][0],batch[-1][0]))

        # All-atom stage (skips relax if -cen_only on)
    
        fa_score = PR.create_score_function("ref2015_cart")
        fa_score.set_weight(PR.rosetta.core.scoring.atom_pair_constraint, self.opt.fa_cst_w) 
        fa_score.set_weight(PR.rosetta.core.scoring.coordinate_constraint, self.opt.fa_cst_w) 
        constraints = PR.rosetta.protocols.constraint_movers.ConstraintSetMover()

        outposes = []
        for modelno,(score,pose) in enumerate(batch):
            if not self.opt.cen_only:
                extra_score = [('score',score)]
            
                gdtmm = -1
                if refpose != None and not self.opt.mute:
                    gdtmm = PR.rosetta.protocols.hybridization.get_gdtmm(refpose,pose,aln)
                    print("GDTMM_cen: %5.3f"%(gdtmm))
                    extra_score.append(('GDTMM_cen',gdtmm))
                
                tag = '%s_%04d.cen.pdb'%(self.opt.prefix,modelno)
                if self.opt.write_pose:
                    censilent = self.opt.outsilent.replace('.out','.cen.out')
                    report_pose(pose,tag,extra_score,censilent,self.opt.mute)
                
                # maybe short-prerelax with cencst here to prevent blow-up?
                if self.opt.fa_cst != None:
                    constraints.constraint_file(self.opt.fa_cst)
                    constraints.apply(pose)

                rosetta_utils.relax(pose,fa_score)
                if self.opt.verbose:
                    fa_score.show(pose)
                else:
                    # not dumping log?
                    #fa_score.show_line_headers(T)
                    #fa_score.show_line(T, pose)
                    pass
                
                score = fa_score.score(pose)
            
            extra_score = [('score',score)]
            gdtmm = -1
            if refpose != None and not self.opt.mute:
                gdtmm = PR.rosetta.protocols.hybridization.get_gdtmm(refpose,pose,aln)
                print("GDTMM_final: %5.3f"%(gdtmm))
                extra_score.append(('GDTMM_final',gdtmm))
            
            tag = '%s_%04d.pdb'%(self.opt.prefix,modelno)
            if self.opt.write_pose: report_pose(pose,tag,extra_score,self.opt.outsilent)

            outposes.append(pose)
            
        return outposes
        
if __name__ == "__main__":
    a = Runner()
    a.apply()
