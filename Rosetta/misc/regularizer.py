import copy,random,math,sys
import pyrosetta as PR #call explicitly
import SamplingOperators as SO
import rosetta_utils
import numpy as np
#sys.path.insert(0,'/home/hpark/projects/ML/pyhyb/merge/scripts')
#from myutils import list2part, trim_lessthan_3
sys.path.insert(0,'/home/hpark/util3')
from misc import list2part, trim_lessthan_3

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
    opt.add_argument('-ss_fn', dest='ss_fn', metavar="SS_FN", required=True,
                     help='SS prediction file')

    ## Output
    opt.add_argument('-prefix', dest='prefix', default="S", \
                     help='Prefix of output pdb file')
    opt.add_argument('-nstruct', dest='nstruct', type=int, default=1,
                     help='number of structures to sample serially')
    opt.add_argument('-debug', dest='debug', default=False, action='store_true', \
                     help='Debug mode [turn off]')
    opt.add_argument('-verbose', dest='verbose', default=False, action='store_true', \
                     help='Verbose mode [turn off]')
    opt.add_argument('-mute', dest='mute', default=False, action='store_true', \
                     help='Mute all [turn off]')
    opt.add_argument('-kT',          dest='kT0',          type=float, default=  2.0,
                     help="MC temperature factor")

    #
    if len(argv) <= 1:
        opt.print_help()
        sys.exit(1)

    # support for Rosetta style flags file
    for arg in copy.copy(argv):
        if arg[0] == '@':
            if not os.path.exists(arg[1:]):
                sys.exit('Flags file "%s" does not exist!'%arg[1:])
            for l in open(arg[1:]):
                if l.startswith('#'): continue
                argv += l[:-1].strip().split()
            argv.remove(arg)
            
    params = opt.parse_args(argv)
    params.cen_cst_w = 1.0
    params.frag_insertion_mode = "random"

    return params

def apply_crd_cst(pose):
    nres = pose.size()
    for ires in range(nres):
        if pose.residue(ires+1).is_virtual_residue(): continue
        atm = 'CA'
        atomid_i = PR.rosetta.core.id.AtomID( pose.residue(ires+1).atom_index(atm), ires+1 )
        xyz = pose.xyz(atomid_i)
        
        fx = PR.rosetta.core.scoring.func.HarmonicFunc( 0.0, 1.0 ) #sigma=1.0!
        cst = PR.rosetta.core.scoring.constraints.CoordinateConstraint( atomid_i, atomid_i, xyz, fx )
        pose.add_constraint( cst )
    
# One step of MC in FT hybridize
def MC(pose,scorer,sampler,tot_it,
       recover_min=True,recover_min_every=1e6,
       verbose=False):
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
        sampler.apply(pose_work)
            
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
            print( "k/score: %4d %5.1f %9.3f %9.3f %9.3f %5.3f"%(it,kT,E,Eprv,Emin,accratio) )
        if accepted: Eprv = E
        
    if recover_min: pose = pose_min
    return pose, accratio

# Reproduced FT hybridize in Rosetta
def FoldTreeHybridize(pose,opt,refpose=None,mute=False):
    nres = pose.size()
    pose0 = pose.clone()

    # 0. Score setup: term weights adjusted by scheduler
    scorer = Scorer(opt) 

    strand_candidates = []
    for i in range(pose.size()):
        PE = scorer.ss3[i][1]
        if PE > 0.8: strand_candidates.append(i+1)
    strand_candidates = list2part(trim_lessthan_3(strand_candidates,pose.size()))
    for strand in strand_candidates:
        print( "Extra SS from input: ", strand )
    
    ## 1. FoldTree setup
    cuts,jumps = rosetta_utils.setup_fold_tree(pose,[],#min_chunk_len=[2,5,1000],
                                               additional_SS_def=strand_candidates,
                                               report=opt.verbose)

    jump_anchors = [jump[1] for jump in jumps]
    alned_res = list(range(1,nres+1))

    if opt.debug: pose.dump_pdb("ftsetup.pdb")

    ## 2. Sampling operators
    # originally fragment insertion prohibitted at jump_anchors
    # -- should revisit?? (or, instead on cuts)
    disallowed = jump_anchors #cuts
    if not mute: print( "Fragment insertion disallowed: "+" %d"*len(disallowed)%tuple(disallowed))

    opt.aligned_frag_w = 1.0 #everywhere

    # Regular Mutation operator with big-frag insertion
    mutOperator1 = SO.FragmentInserter(opt,opt.fragbig,nres,alned_res,disallowed,
                                       name ="FragBig")
    # Regular mutation operator with small-frag insertion
    mutOperator2 = SO.FragmentInserter(opt,opt.fragsmall,nres,alned_res,disallowed,
                                       name="FragSmall")

    perturber = SO.SamplingMaster(opt,[mutOperator1],[1.0],"perturber")
    inserter = SO.SamplingMaster(opt,[mutOperator1,mutOperator2],
                                 [1.0,0.0],"inserter") #probability
    refiner = SO.SamplingMaster(opt,[mutOperator2],[1.0],"refiner")

    apply_crd_cst(pose)

    if not mute: print( "Starting score: %8.3f"%scorer.score(pose))

    # 5. MC
    scheduler = Scheduler(opt)
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
        pose,accratio = MC(pose,scorer,sampler,niter,verbose=opt.verbose)
        chain_brk = scorer.get_term(pose, PR.rosetta.core.scoring.linear_chainbreak)
        
        l = "Stage%d (Acceptance %5.3f): %8.3f -> %8.3f, chainbrk: %8.3f"%(it+1,accratio,score0,scorer.score(pose),chain_brk)
        if scorer.Edssp < 0:
            l += ', DSSP %8.3f'%scorer.Edssp

        gdtmm = PR.rosetta.core.scoring.CA_gdtmm(pose0,pose)
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
        sfxn.set_weight(PR.rosetta.core.scoring.hbond_lr_bb, 5.0 ) 
        sfxn.set_weight(PR.rosetta.core.scoring.rama_prepro, 3.0 ) 
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
        if opt.debug:
            self.niter = [200,  10,   10, 500]
            self.kT    = [2.5,  1.0, 1.0, 1.0]
            self.wvdw  = [0.1,  1.0, 1.0, 1.0]
            self.wbrk  = [0.0,  0.1, 0.5, 1.0]
            self.wcst  = [0.0,  0.1,0.25, 1.0]

        self.kT   = [opt.kT0*kT for kT in self.kT]
        self.wcst = [w*opt.cen_cst_w for w in self.wcst]

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
    def __init__(self):
        self.opt = arg_parser(sys.argv[1:])
            
    def apply(self):
        pose0 = PR.pose_from_file(self.opt.pdb_fn)
        PR.SwitchResidueTypeSetMover("centroid").apply(pose0)

        # store original FT
        ft = pose0.conformation().fold_tree().clone()

        nsample_cen = self.opt.nstruct

        # Repeat Centroid-MC nsample_cen times
        fa_score = PR.create_score_function("ref2015_cart")
        fa_score.set_weight(PR.rosetta.core.scoring.coordinate_constraint, 5.0 ) 
        fa_score.set_weight(PR.rosetta.core.scoring.hbond_lr_bb, 5.0 ) 
        fa_score.set_weight(PR.rosetta.core.scoring.fa_elec, 3.0 ) 
        fa_score.set_weight(PR.rosetta.core.scoring.rama_prepro, 3.0 ) 

        outposes = []
        for modelno in range(nsample_cen):
            pose = pose0.clone()
            if not self.opt.mute: print("Generating %d/%d structure..."%(modelno+1,nsample_cen))
            # Coarse-grained sampling stage
            pose,score = FoldTreeHybridize(pose,self.opt,pose0,self.opt.mute) 

            # recover original fully-connected FT
            rosetta_utils.reset_fold_tree(pose,pose.size()-1,ft)
            
            rosetta_utils.relax(pose,fa_score)
            
            gdtmm = PR.rosetta.core.scoring.CA_gdtmm(pose0,pose)
            print("Centroid score: %8.3f, GDTMM %5.3f"%(score,gdtmm))
            
            tag = '%s_%04d.pdb'%(self.opt.prefix,modelno)
            pose.dump_pdb(tag)
            outposes.append(pose)
            
        return outposes
        
if __name__ == "__main__":
    a = Runner()
    a.apply()
