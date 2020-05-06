import sys,copy,glob,os
import pyrosetta as PR

import basic.config #as config
import basic.rosetta_utils #as rosetta_utils
from basic.PoseInfo import PoseInfo
from basic.FoldTreeInfo import FoldTreeInfo

from ChunkSampler.Matcher import Matcher
from ChunkSampler.TERM import TERMDB

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SCRIPTDIR+'/../../utils')
import myutils

def arg_parser(argv):
    import argparse
    parser = argparse.ArgumentParser()

    ## Input
    parser.add_argument('-pdb', dest='pdb', required=True, help='Input pdb file')
    parser.add_argument('-SSpred', dest='SSpred_fn', required=True,
                     help="SS prediction results as npz format")
    
    parser.add_argument('-npz', dest='npz', required=True,
                     help='DeepAccNet results as npz format')

    ## output
    parser.add_argument('-outprefix', default="FT",
                        help='prefix of output npz file')
    
    # Optional
    parser.add_argument('-offset', dest='offset', default=0.0, help="Offset in filtering score")
    parser.add_argument('-prefix', dest='prefix', default="t000_",
                        help="Prefix of output")
    parser.add_argument('-cut_at_end', dest='cut_at_mid', default=True, action="store_false",
                        help="define cut at the end of loop, otherwise at the mid of the loop")
    parser.add_argument('-do_subdef', dest='do_subdef', default=False, action="store_true",
                        help="Use subgraph decompo to get variable Jump connectivity outputs")
    parser.add_argument('-do_termsearch', dest='do_termsearch', default=False, action="store_true",
                        help="Use TERM db lookup to get variable ULR-SS jump candidates")
    parser.add_argument('-allow_beta_jump', default=True, action='store_true',
                        help="Allow jump DOF sampling of beta strands") # block this at pyRefinQ.py if you want
    parser.add_argument('-simple_mm', default=False, action='store_true',
                        help="generate only 1 mm")
    parser.add_argument('-nojump', default=False, action='store_true',
                        help='disallow jump moves')
    parser.add_argument('-npert_jump_loc', type=int, default=0,
                        help='Number of FTs with jump location perturbations')


    #parser.add_argument('-subdef_confience_offset',

    # debugging
    parser.add_argument('-write_pdb', dest='write_pdb', default=False, action="store_true",
                     help='report library as pdb files')
    parser.add_argument('-debug', dest='debug', action='store_true', default=False )

    if len(argv) == 1:
        parser.print_help()
        return

    opt = parser.parse_args()
    opt.config = basic.config.CONFIGS
    return opt

# unused
def idealize_ulrs(pose,poseinfo,FTInfo):
    # EXTRA: Idealize & extend at ULR regions so that torsion replacement make sense...
    # CHECK THROUGH ESTOGRAM && CONFIDENCE
    print("\n[ChunkSampler/main] ======Extend and idealize pose around ULR=======")
    pose_ext = pose
    stoppoints = FTInfo.cuts + [min(jump.reslist) for jump in FTInfo.jumps_nonulr] +\
                 [max(jump.reslist) for jump in FTInfo.jumps_nonulr]
    stoppoints.sort()
    extended_mask = [False for res in range(poseinfo.nres)]
    for ulr in FTInfo.ulrs:
        basic.rosetta_utils.local_extend(pose_ext,extended_mask,ulr,stoppoints,idealize=True)
    if opt.debug: pose_ext.dump_pdb("ideal_ext.pdb")
    poseinfo.extended_mask = extended_mask #store at poseinfo
    return pose_ext

def TERMsearch(opt,refpose,poseinfo,FTInfo):
    # TERM database; TODO: make loop db in same format
    full_db = {}
    for SStype in opt.config['SSTYPE_SUPPORTED']:
        db = TERMDB(opt,SStype)
        if db.db != []: full_db[SStype] = db
        
    anctypes = [SS[:-1] for SS in opt.config['SSTYPE_SUPPORTED']]

    #precompute potential ULR anchors
    ulrSSs = FTInfo.SSs_ulr
    poseinfo.find_ULR_anchors(ulrSSs,anctypes,report=opt.debug)

    # get maximum closable resrange 
    stoppoints = FTInfo.cuts + [SS.begin for SS in FTInfo.SSs_reg] +\
                  [SS.end for SS in FTInfo.SSs_reg]
    stoppoints.sort()
    extended_mask = [False for res in range(poseinfo.nres)]
    for ulr in FTInfo.ulrs:
        basic.rosetta_utils.local_extend(poseinfo.nres,extended_mask,ulr,stoppoints)
    poseinfo.extended_mask = extended_mask
    
    # build clash grid
    ulrres = []
    for ulr in FTInfo.ulrs_aggr: ulrres += ulr
    poseinfo.build_clash_grid(ulrres)

    ### SCAN STARTS
    ##  Scan through SScombs 
    print("\n[ChunkSampler/main] ========== Scan %d ULRSSs =========="%len(ulrSSs))
    for iulr,ulrSS in enumerate(ulrSSs): 
        print("\n** Search ULRSS %d-%d"%(ulrSS.begin,ulrSS.end))
        # storage for iulr-th ULRSS (each Matcher.MatchSolutions class)
        solutions_tot = []
        skipped_SScomb = []
        for SScomb in full_db:
            # Convention: last SStype corresponds to ULR pred. SStype
            if SScomb[-1] != ulrSS.SStype:
                skipped_SScomb.append(SScomb)
                continue
            
            # 5a. Setup a matcher for certain SS combination
            db = full_db[SScomb] #TERMDB class
            matcher = Matcher( db, opt, prefix=SScomb, debug=opt.write_pdb )
            #rmsdcut_from_init=rmsdcut ) #controls b/w refine<->aggressive
            print( "\n----Searching through TERM DB %s with %d candidates"%(SScomb,len(db.db)))

            # 5b. Search against db -- aggressive search
            SSanchors = poseinfo.relevant_ULRanchors_for_SScomb(SScomb)
            solutions = matcher.place_ULR_at_anchors( poseinfo, SSanchors, ulrSS )
                
            nthreads = sum([len(sol.threads) for sol in solutions])
            print( " - Num. solutions/threads from %s at ULR %d-%d: %d/%d"%(SScomb,ulrSS.begin,ulrSS.end,
                                                                            len(solutions),nthreads) )

            solutions_tot += solutions

        print( " - Skipped DB: ", " ".join(skipped_SScomb) )

        # 5c. Convert solution info to Rosetta jump class
        # FoldTreeInfo.Jump type

        # this call-by-ref!
        jump_at_ULR = FTInfo.get_jump(ulrSS.reslist)
        jumpid = FTInfo.get_jumpid(ulrSS.reslist)
        #jump_at_ULR = FTInfo.jumps[ulrSS.jumpid] # check...
        if not jump_at_ULR:
            print("Jump not found for ", ulrSS.reslist)
            continue
        
        for i,solution in enumerate(solutions_tot): 
            ULRSSframe = solution.SSs[-1]
            #solution.write_as_pdb("part."+solution.tag+".pdb")

            # Interface to Rosetta Jump!
            #threads = [sol[2] for sol in solution.threads] #sol = (begin,end,cenpos)
            #threads = [1,2,3] #+- 1 seq
            threads = [2]
            jump_at_ULR.append_moveset_from_match(refpose,solution,threads,iSS=-1)

            '''
            #oldway
            newjumps,stubs,ijump = rosetta_utils.SS_to_jump( pose_ext,
                                                             ULRSSframe, #frame info from solution
                                                             FTInfo,
                                                             threads )
            '''

def get_list_of_ulr_defs(FTInfo):
    # take full ulr
    defs = [FTInfo.ulrs_aggr,FTInfo.ulrs_cons]

    # take one by one
    for ulr in FTInfo.ulrs_aggr:
        defs.append([ulr])

    # see if any other jump is less reliable...
    for i,jump in enumerate(FTInfo.jumps):
        jumpconf = np.mean(self.Qres[jump.reslist[0]-1:jump.reslist[-1]]) #confidence of current struct
        print( " Jump %d %3d-%3d (ulr=%d): confidence %.3f"%(i,jump.reslist[0],jump.reslist[-1],
                                                             jump.is_ULR,jumpconf) )
                            
    return defs 
    
def main(opt,pose=None,FTnpz=None):
    if pose == None and opt.pdb != None:
        pose = PR.pose_from_pdb(opt.pdb)
    if FTnpz == None:
        FTnpz = opt.npz

    if pose == None or FTnpz == None:
        sys.exit("No input pose and/or FTnpz provided!")
    
    ### SETUP PART
    # 1. Pose info setup
    poseinfo = PoseInfo(pose,opt,SSpred_fn=opt.SSpred_fn)
    
    # 2. FoldTree setup
    # read in ULR/Jump info from estogram2FT: estogram should be always provided
    
    FTInfo = FoldTreeInfo(pose,opt)
    FTInfo.init_from_estogram(pose,opt,npz=FTnpz,poseinfo=poseinfo)
    
    #ulrSSs = FTInfo.extraSS_at_ulr # Defined as list of SSclass, ULR but predicted as SS
    variable_ulr_defs = get_list_of_ulr_defs(FTInfo)

    ## Output multiple FT options of ulrs/subs into individual npz
    if opt.simple_mm:
        FTInfo.setup_fold_tree(pose,opt,poseinfo,ulrs=FTInfo.ulrs_cons)
        FTInfo.save_as_npz(opt.outprefix+".simple.npz",fQcut=0.2)
        return

    # a. freeze sub & go through ulrs
    for i,ulrs in enumerate(variable_ulr_defs[:2]):
        print("======= Generating %s.ulrdef%d.npz..."%(opt.outprefix,i))
        FTInfo.setup_fold_tree(pose,opt,poseinfo,ulrs=ulrs)
        if opt.do_termsearch:
            TERMsearch(opt,pose,poseinfo,FTInfo)
        fQcut = 0.2
        if opt.nojump: fQcut = 0.0
        FTInfo.save_as_npz(opt.outprefix+".ulrdef%d.npz"%(i),fQcut=fQcut)
        
    # b. allow jumps to move, freeze ulr as
    if not opt.nojump: 
        FTInfo_tmp = copy.copy(FTInfo) #safe???
        FTInfo_tmp.setup_fold_tree(pose,opt,poseinfo,ulrs=FTInfo.ulrs_cons)
        for i,fQcut in enumerate([0.2,0.4,1.0]):
            print("======= Generating %s.jump%d.npz..."%(opt.outprefix,i))
            FTInfo_tmp.save_as_npz(opt.outprefix+".jump%d.npz"%(i),fQcut=fQcut)

    # c. freeze ulrs & go through subs
    '''
    for i,subdef in enumerate(FTInfo.subdefs):
        print("======= Generating %s.subdef%d.npz..."%(opt.outprefix,i))
        FTInfo.setup_fold_tree(pose,opt,poseinfo,subdef_in=subdef)
        FTInfo.save_as_npz(opt.outprefix+".subdef%d.npz"%(i))
    '''

    # d. randomize jump centers a bit
    '''
    for i in range(opt.npert_jump_loc):
        print("======= Generating %s.cendef%d.npz..."%(opt.outprefix,i))
        FTInfo_tmp = copy.copy(FTInfo) #safe???
        FTInfo_tmp.setup_fold_tree(pose,opt,poseinfo,ulrs=ulrs,max_pert_jump_loc=3)
        FTInfo_tmp.save_as_npz(opt.outprefix+".cendef%d.npz"%(i))
    '''

    # e. term search
    #if opt.do_termsearch: # append info into FTInfo
    #    FTInfo.setup_fold_tree(pose,opt,poseinfo,ulrs=FTInfo.ulrs_cons)
    #    TERMsearch(opt,poseinfo,FTInfo)
    
# check
def check(opt):
    npzs = glob.glob(opt.outprefix+".cendef?.npz")
    for npz in npzs:
        print("*** Loading %s"%npz)
        pose = PR.pose_from_pdb(opt.pdb)
        FTInfo = FoldTreeInfo(pose,opt)
        FTInfo.load_from_npz(npz,pose,report=True)
        
if __name__ == "__main__":
    opt = arg_parser(sys.argv[1:])
    PR.init('-mute all')
    main(opt)
    check(opt)
