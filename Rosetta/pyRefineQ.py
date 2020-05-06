import sys,os
import copy,random
import numpy as np
import time
import pyrosetta as PR #call explicitly
from Scorer import Scorer

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,SCRIPTDIR+'/basic')
import SamplingOperators as SO
import rosetta_utils
import estogram2cst
from FoldTreeInfo import FoldTreeInfo
import config

sys.path.insert(0,SCRIPTDIR+'/../Critics')
from FaQRunner import FaQRunner

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
    #opt.add_argument('-estogram', dest='estogram', default=None, help='Outcome of DeepAccNet in npz format')
    opt.add_argument('-FT', dest='fts', default=[], nargs="+",
                     help='FoldTree definitions')
 
    
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
    opt.add_argument('-score', dest='scoretype', metavar="SCORE", default="default",\
                     help='centroid Score type')
    opt.add_argument('-cen_cst', dest='cen_cst', default=None, help='centroid restraint file')
    opt.add_argument('-fa_cst', dest='fa_cst', default=None, help='fullatom restraint file')
    opt.add_argument('-cen_cst_w', dest='cen_cst_w', type=float, default=1.0, help='')
    opt.add_argument('-fa_cst_w', dest='fa_cst_w', type=float, default=1.0, help='')
    opt.add_argument('-refcorrection_method', dest='refcorrection_method', default="none", help='')
    opt.add_argument('-Pspline_on_fa', type=float, default=0.3, help="Lower bound of max(P) applying splineas cst") 
    opt.add_argument('-hardcsttype', default="sigmoid", help="functional form of core restraints")
    opt.add_argument('-dynamic_Pcut', default=False, action="store_true", \
                     help="apply dynamic Pcore cut to cover sufficient num. pairs")
    opt.add_argument('-Pcore', default=0.8, type=float, help="estogram confident pair starting Pcen") 
    opt.add_argument('-dist', dest='dist', default=None, help='Outcome of trRosetta in npz format which contains predicted distogram. It should be provided if you use Q as score')

    ## Macro cycle options
    opt.add_argument("-relaxscript", dest="relaxscript",
                     default="%s/misc/faster.script"%SCRIPTDIR)
    
    ###### Sampling options
    ### Scheduling
    opt.add_argument('-nstep_anneal', dest="nstep_anneal", type=int, default=15,
                     help='num. steps for each AnnealQ runs') #deprecated
    opt.add_argument('-nmacro', type=int, default=1,
                     help='num. steps for macro cycles')
    opt.add_argument('-nbranch', type=int, default=50,
                     help='nsamples to branch every step')
    
    ### relative weights
    opt.add_argument('-mover_weights', dest='mover_weights', nargs='+', type=float, default=[], #auto
                     help="weights on movers [frag-insert/jump-opt/unused]")
    opt.add_argument('-branch_mover', default=False, action="store_true",
                     help="Apply same type of mover across all the branching samples")
    
    ### Jump-opt
    ## 1. Through Auto setup: no input args for ulr / sub_def
    opt.add_argument('-autoFT', dest='autoFT', default=False, action='store_true', \
                     help='automatically setup FT looking at estogram')
    opt.add_argument('-termsearch_mode', default="major",
                     help='which jump to try term (default=major: try only least confident)')

    ## 2. or Through extra user-provided arguments 
    opt.add_argument('-ulr', dest='ulr_s', metavar='ULRs', nargs='+', default=[], \
                     help='ULRs should be sampled. (e.g. 5-10 16-20). If not provided, whole structure will be sampled')
    opt.add_argument('-sub_def', dest='sub_def', nargs='+', default=[],
                     help="definition of subs: argument same as -ulr")
    opt.add_argument('-allow_beta_jump', default=False, action='store_true',
                     help="Allow jump DOF sampling of beta strands") # False by default
                      
    ### FragInsert
    opt.add_argument('-keep_ulr_tors', default=False, action='store_true',
                     help='preserve ulr torsion angles at the initial FT setup')
    opt.add_argument('-fragopt', default='single',
                     help='Fragment insertion type (single/mc)')
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
    params.config = config.CONFIGS
    
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
    params.Pcore = [params.Pcore,params.Pcore,params.Pcore+0.1]

    # check if any 
    check_opt_consistency(params)
        
    return params

def check_opt_consistency(opt):
    return True

##### simple function using GPU 
def branch_and_select_1step(pose_in,sampler,scorer,n,
                            minchange=0.0, # for aggressive movement
                            minimizer=None,
                            branch_mover=False,
                            prefix=None,
                            #kT=0.1,
                            #kT=99 #always drift (i.e. No reject),
):
    time0 = time.time()
    poses = []
    rmsds = []

    # pre-select mover
    movername = 'random'
    if branch_mover: movername = sampler.select_mover()
    
    for i in range(n):
        pose = pose_in.clone()
        move = sampler.apply(pose,move=movername)
        rmsd = PR.rosetta.core.scoring.CA_rmsd(pose,pose_in)
        rmsds.append(rmsd)
        poses.append(pose)
        if prefix != None: pose.dump_pdb(prefix+"%02d.pdb"%i)

    # append starting pose for scoring
    #poses.append(pose_in)

    time1 = time.time()
    scores = scorer.score(poses)
    time2 = time.time()

    sort_by_score = [i for i in np.argsort(scores) if rmsds[i] > minchange]
    if len(sort_by_score) == 0:
        sort_by_score = np.argsort(scores)

    print(' '.join(["%3.1f"%rmsds[i] for i in sort_by_score[:min(20,len(sort_by_score))]]))

    imin = sort_by_score[0]
    
    Ecomp_min = scorer.by_component[imin] #cached
    extralog = "Sel %02d, Elapsed time, pert(min)/score: %.1f %.1f"%(imin,time1-time0,time2-time1)
    extralog += "; Moveset %s"%movername
    extralog += "; Emin comp: "+' '.join(["%s %6.3f"%(key,Ecomp_min[key]) for key in Ecomp_min])
    
    return poses[imin], scores[imin], extralog

class Scheduler:
    def __init__(self,opt,do_Qopt=False):
        # leave it undefined unless overloaded below
        self.wQ = []
        self.wdssp = [] 
        self.nstep_anneal = opt.nstep_anneal

        if do_Qopt:
            print("Scheduler: Loading Qanneal dflt schedule.")
            self.load_Qanneal_sch()
        elif opt.debug:
            print("Scheduler: Loading debug schedule.")
            self.load_debug_sch()
        else:
            print("Scheduler: Loading Hybrid dflt schedule.")
            self.load_dflt_sch()
        #apply as "scale"
        self.kT   = [opt.kT0*kT for kT in self.kT]
        self.wcst = [w*opt.cen_cst_w for w in self.wcst]

    def load_dflt_sch(self):
        # define a default schedule
        #originally in hybrid
        # stage1: score0, vdw0.1 brk0  -> 1000
        # stage2: score1, vdw1   brk0.1 cst0.1 -> 1000
        # stage3: {score2<->score5}, vdw1  brk0.25 cst0.25 -> 10*1000
        # stage4: score3, full scale -> 3*200
        self.niter = [  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  25]
        self.kT    = [2.5,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.wvdw  = [0.1,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.wbrk  = [0.0,  0.1, 0.1, 0.3, 0.2, 0.7, 0.5, 1.0, 0.8, 1.0, 1.0]
        self.wcst  = [0.0,  0.1,0.25,0.25,0.25,0.25,0.25,0.25, 0.25,0.25,1.0]

    def load_Qanneal_sch(self):
        #pert/insert/refine/closure
        self.niter = [  10,  self.nstep_anneal, 10, 6] #pert/Qanneal/Qrefine/closure
        #self.niter = [   10,   0,   0,  10] #DEBUG
        self.kT    = [  0.0, 0.0, 0.0, 1.0]
        #these values are scales wrt input wts
        self.wvdw  = [  0.1, 1.0, 1.0, 1.0]
        self.wbrk  = [  0.0, 0.1, 0.1, 1.0] #keep nonzero for reporting purpose
        self.wcst  = [  0.1, 1.0, 1.0, 1.0]
        self.wQ    = [  0.1, 1.0, 1.0, 0.0]
        self.wdssp = [  0.1, 0.1, 1.0, 1.0]

    def load_debug_sch(self):
        self.niter = [100 , 500, 500, 500]
        self.kT    = [2.5,  2.5, 2.5, 2.5]
        self.wvdw  = [0.1,  1.0, 1.0, 1.0]
        self.wbrk  = [0.0,  0.1, 0.5, 1.0]
        self.wcst  = [0.0,  0.1,0.25, 1.0]

    def nstages(self):
        return len(self.niter)
        
    def adjust_scorer_scale(self,it,scorer):
        scorer.reset_kT( self.kT[it] )
        if it < len(self.wQ): scorer.reset_scale( "Qcen", self.wQ[it] )
        if it < len(self.wdssp): scorer.reset_scale( "dssp", self.wdssp[it] )
        scorer.reset_scale( PR.rosetta.core.scoring.vdw, self.wvdw[it] )
        scorer.reset_scale( PR.rosetta.core.scoring.linear_chainbreak,    self.wbrk[it] )
        scorer.reset_scale( PR.rosetta.core.scoring.atom_pair_constraint, self.wcst[it] )
        print("Adjusted weight: ", scorer.wts)
        
    def get_niter(self,it):
        return self.niter[it]

class Annealer:
    def __init__(self,opt):
        self.opt = opt
    
    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        pass

    def initialize_pose_with_FT(self,pose,FTInfo,relax=True):
        # first make as extended at ulr
        if not self.opt.keep_ulr_tors:
            ulrres = []
            for ulr in FTInfo.ulrs: ulrres += list(ulr)
            for resno in ulrres:
                pose.set_phi(resno,-135.0)
                pose.set_psi(resno, 135.0)
                pose.set_omega(resno,180.0)

        # check whether TERMdb declared in FTInfo; if not, pass
        #weight = np.array([len(jump.movesets) for jump in FTInfo.jumps],dtype='float16')
        weight = np.zeros(len(FTInfo.jumps))
        for i,jump in enumerate(FTInfo.jumps):
            if len(jump.movesets) > 0: weight[i] = 1.0/(jump.Q+0.01)
        
        if sum(weight) > 0:
            weight /= float(np.sum(weight))
            if self.opt.termsearch_mode == "major":
                jumpid = np.argmax(weight)
            else:
                jumpid = random.choices(list(range(len(FTInfo.jumps))),weight)[0]
            print("TERM search weight: ", weight, "selection opt=%s, jumpid: %d"%(self.opt.termsearch_mode,jumpid))

            # apply and select
            jump = FTInfo.jumps[jumpid]

            sfxn = PR.create_score_function("score0.wts") #vdw 0.1
            sfxn.set_weight( PR.rosetta.core.scoring.atom_pair_constraint, 0.1 )
            sfxn.set_weight( PR.rosetta.core.scoring.cen_hb, 0.1 ) #bias to parallel?
            sfxn.set_weight( PR.rosetta.core.scoring.hbond, 0.1 )

            Es = []
            poses = []
            for k in range(50): #randomly pick from N: higher N, less diversity (as the same thing will be picked always)
                pose_,restags = jump.moveset_to_pose(pose,mode='random',report_pdb=False)
                pose = pose_[0]
                restag = restags[0]
            
                sfxn.score(pose)
                Ev = pose.energies() 
                Esum = sum([Ev.residue_total_energy(res) for res in restag])
                Es.append(Esum)
                #pose.dump_pdb("jump%d.pdb"%k)
                poses.append(pose)
                
            # pick lowest E
            pose = poses[np.argsort(Es)[0]]
            print(np.argsort(Es))
            print(Es)
        pose.dump_pdb('switch.pdb')

        # only if initially extended...
        if relax and not self.opt.keep_ulr_tors:
            residue_weights = np.zeros(FTInfo.nres)
            for res in ulrres:
                insertable = [res+k for k in range(9) if res+k in ulrres] #allow +-2 change
                if len(insertable) > 7: residue_weights[res-1] = 1.0
            multiFragInsertor = SO.FragmentMC(self.opt.fragbig,
                                              residue_weights,
                                              mciter=100,kT=1.0)
            pose = multiFragInsertor.apply(pose)

        return pose
    
    def apply(self,pose0,FTnpz,
              runno=0,report_trj=""):
        # pose0 can be provided through argument or optparser
        PR.SwitchResidueTypeSetMover("centroid").apply(pose0)

        # Get native pose & seqaln to it
        refpose = None
        if self.opt.native != None:
            refpose = PR.pose_from_file(self.opt.native)
            
        # store original FT
        nres = pose0.size()
        ft0 = pose0.conformation().fold_tree().clone()

        ## FTsetup: retreive FT from input npz file
        pose = pose0.clone()
        FTInfo = FoldTreeInfo(pose,self.opt) #opt should have --- ?

        if FTnpz == None:
            rosetta_utils.simple_ft_setup(pose,ulrs=self.opt.ulrs)
        else:
            FTInfo.load_from_npz(FTnpz,pose,report=True)

        scorer = Scorer(self.opt, FTInfo.cuts, normf=1.0/pose.size())

        ## Score/Cst setup
        if self.opt.scoretype == "default":
            self.opt.scoretype = "Q"

        if self.opt.cen_cst.endswith('npz'): # from DL
            self.opt.Pcontact_cut =1.1 #==None
            self.opt.refcorrection_method="none"  #"statQ"
            # below: through input arg
            #self.opt.Pcore = [0.8,0.8,0.9] # 
            #self.opt.dynamic_Pcut = False
            #self.opt.hardcsttype="sigmoid" #soft for only very confident ones
            #self.opt.Pspline_on_fa=0.3 #==throw away if maxP in estogram lower than this val

            # refpose == pose0
            print("ULR?", FTInfo.ulrs)
            ulrres = []
            for ulr in FTInfo.ulrs: ulrres += list(ulr)
            estogram2cst.apply_on_pose( pose,npz=self.opt.cen_cst,
                                        opt=self.opt,
                                        mask=ulrres )

        # estogram used to filter jumps
        pose = self.initialize_pose_with_FT(pose,FTInfo)
        pose.dump_pdb('iFT.pdb')

        ## Sampler --  FTInfo-aware version
        print("Get samplers...")
        perturber, inserter, refiner, closer = SO.get_samplers(pose,self.opt,FTInfo)
        scheduler = Scheduler(self.opt,do_Qopt=('Qcen' in scorer.wts))
        
        ## Coarse-grained sampling stage
        # 1. perturb initially
        print("Perturb...")

        poses = []
        for i in range(self.opt.nbranch):
            pose_work = pose.clone()
            for k in range(scheduler.niter[0]):
                perturber.apply(pose_work)
            poses.append(pose_work)

        scheduler.adjust_scorer_scale(0,scorer)
        Es = scorer.score(poses)
        imin = np.argsort(Es)[0]
        pose = poses[imin]

        pose_min = pose
        Emin = Es[imin]
        pose_min.dump_pdb("ipert.pdb")
        print("Einit ",Emin)

        # to report
        poses_out = []
        # 2. Run sampling stages
        it = 0
        for stage in range(1,scheduler.nstages()):
            niter = scheduler.get_niter(stage)
            print("Running stage %d, niter=%d..."%(stage,niter))
            scheduler.adjust_scorer_scale(stage,scorer)
            minchange=0.0         
            if stage == scheduler.nstages()-2:
                sampler = refiner

            elif stage == scheduler.nstages()-1:
                closer.kT = scheduler.kT[stage]
                sampler = closer
            else:
                sampler = inserter
                minchange = 1.0 #has to move at least X Ang every step

            sampler.show_ops()
                        
            for i in range(niter):
                # now that scorer weights readjusted, let's renew Emin
                if i == 0: Emin = scorer.score(pose_min)

                pose_in = pose_min
                pose_out, Eout, extrainfo \
                    = branch_and_select_1step(pose_in,sampler,scorer,
                                              self.opt.nbranch,
                                              minchange=minchange,
                                              branch_mover=self.opt.branch_mover,
                                              #prefix="tmp.%d."%(it)
                    )

                # report insertion sites
                if self.opt.debug: sampler.show()
            
                print("ANNEAL: %3d %7.4f %7.4f %1d | %s"%(i,Emin,Eout,(Eout<Emin),extrainfo))
                if Eout < Emin: #annealing
                    pose_min = pose_out
                    Emin = Eout

                # trajectory
                if it%self.opt.dump_trj_every==0 and it>0:
                    poses_out.append(pose_min)

                    if report_trj == "": continue
                    rosetta_utils.report_pose(pose_min,
                                              tag=self.opt.prefix+".%d.%03d"%(runno,it),
                                              extra_score = [("score",Emin)],
                                              outsilent=report_trj,
                                              refpose=refpose)
                it += 1
                
        scorer.close()
        
        # recover original fully-connected FT
        for i,pose in enumerate(poses_out):
            rosetta_utils.reset_fold_tree(pose,FTInfo.nres,ft0)
            #pose.remove_constraints() #estogram -- decide later
            poses_out[i] = pose #necessary?
        return poses_out
    
class Runner:
    def __init__(self,params=None):
        if params != None:
            self.opt = params
        else:
            self.opt = arg_parser(sys.argv[1:])
            
    def MacroCycle(self,pose0=None,dump_pdb=False):
        if pose0 == None and self.opt.pdb_fn != None:
            pose0 = PR.pose_from_file(self.opt.pdb_fn)

        if pose0 == None:
            sys.exit("No starting pose defined!")
            
        refpose = None
        if self.opt.native != None:
            refpose = PR.pose_from_file(self.opt.native)
    
        #setupFT.add_default_options_if_missing(self.opt)

        Emin = 5.0 #ScorerFA([pose0])
        pose_min = pose0.clone()
 
        #split this for now... will be on-the-fly
        npzs = self.opt.fts #can be None

        # For first shot try nmacro = 1 & nanneal many (>100)
        print("Run %d macrocycles of Qcen-annealing jobs"%(self.opt.nmacro))

        # comment out dump_silent in order to skip trj dumping
        AAscorer = FaQRunner(relax=True,relaxscript=self.opt.relaxscript,
                             verbose=True)

        for it in range(self.opt.nmacro):
            pose = pose_min
            
            # diversify FT on-the-fly
            #npzs = setupFT.main(pose,self.opt)

            # Try annealing with various options
            poses_gen = []
            if len(npzs) == 0:
                npzs_sel = None
            else:
                npz_sel = random.choice(npzs)

            with Annealer(self.opt) as annealer:
                # comment out report_trj in order to skip trj dumping
                poses_out = annealer.apply(pose,npz_sel,
                                           runno=0,
                                           report_trj="%s.trj.cen.out"%self.opt.prefix)

            if self.opt.cen_only:
                print("Skip all-atom stage as -cen_only option provided.")
                pose_min = poses_out[-1]
                continue
            
            # hack to speed up relax
            print("Call FaQRunner...")
            AAscorer.apply(poses_out,'%s/%s'%(os.getcwd(),self.opt.dist),
                           tmppath='tmp.%s'%self.opt.prefix,
                           ncore=min(10,len(poses_out)))
            
            pose_best,Ebest = AAscorer.get_best()

            # dump trj
            AAscorer.dump_silent(outf="%s.trj.out"%self.opt.prefix,
                                 refpose=refpose,
                                 outprefix="iter%02d"%it)
            
            # Metropolis criteria
            if np.exp(-self.opt.config['beta']*(Ebest-Emin)) > random.random():
                pose_min = pose_best
                Emin = Ebest

        if dump_pdb:
            pose_min.dump_pdb(self.opt.prefix+".final.pdb")
        return pose_min
    
if __name__ == "__main__":
    a = Runner()
    a.MacroCycle(dump_pdb=True)
