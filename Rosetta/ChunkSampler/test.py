import sys,copy,glob,os
import pyrosetta as PR

import config
from Matcher import Matcher
from TERM import TERMDB

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0,SCRIPTDIR+'/../basic')
import estogram2FT
from PoseInfo import *
import rosetta_utils

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
    
    # Optional
    parser.add_argument('-offset', dest='offset', default=0.0, help="Offset in filtering score")
    parser.add_argument('-prefix', dest='prefix', default="t000_",
                        help="Prefix of output")

    # debugging
    parser.add_argument('-write_pdb', dest='write_pdb', default=False, action="store_true",
                     help='report library as pdb files')
    parser.add_argument('-debug', dest='debug', action='store_true', default=False )

    if len(argv) == 1:
        parser.print_help()
        return

    opt = parser.parse_args()
    opt.config = config.CONFIGS
    
    return opt

def main(opt):        
    pose = PR.pose_from_pdb(opt.pdb)
    
    #opt.ulrs = [list(range(22,36))] #manual assignment for debugging...

    ### SETUP PART
    ## FoldTreeInfo, PoseInfo, TERMDB, 
    # 1. FoldTree setup:
    # read in ULR/Jump info from estogram2FT: estogram should be always supported
    FTInfo = estogram2FT.main(pose,opt) #== FoldTreeInfo class
    ulrSSs = FTInfo.extraSS_at_ulr # Defined as list of SSclass, ULR but predicted as SS 

    # 2. Pose info setup
    poseinfo = PoseInfo(pose,opt,FTInfo,SSpred_fn=opt.SSpred_fn)
    anctypes = [SS[:-1] for SS in opt.config['SSTYPE_SUPPORTED']]
    poseinfo.find_ULR_anchors(FTInfo,anctypes,report=opt.debug) #precompute potential ULR anchors

    # 3. TERM database; TODO: make loop db in same format
    full_db = {}
    for SStype in opt.config['SSTYPE_SUPPORTED']:
        db = TERMDB(opt,SStype)
        if db.db != []: full_db[SStype] = db

    # EXTRA: Idealize & extend at ULR regions so that torsion replacement make sense...
    # CHECK THROUGH ESTOGRAM && CONFIDENCE
    print("\n[ChunkSampler/main] ======Extend and idealize pose around ULR=======")
    pose_ext = pose
    stoppoints = FTInfo.cuts + [min(jump.reslist) for jump in FTInfo.jumps_nonulr] +\
                 [max(jump.reslist) for jump in FTInfo.jumps_nonulr]
    stoppoints.sort()
    extended_mask = [False for res in range(poseinfo.nres)]
    for ulr in FTInfo.ulrs:
        rosetta_utils.local_extend(pose_ext,extended_mask,ulr,stoppoints,idealize=True)
    if opt.debug: pose_ext.dump_pdb("ideal_ext.pdb")
    poseinfo.extended_mask = extended_mask #store at poseinfo

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
            matcher = Matcher( db, opt, prefix=SScomb, debug=opt.write_pdb,
                               rmsdcut_from_init=2.5 ) #controls b/w refine<->aggressive
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
        jump_at_ULR = FTInfo.get_jump(ulrSS.reslist)
        #jump_at_ULR = FTInfo.jumps[ulrSS.jumpid] # check...
        
        for i,solution in enumerate(solutions_tot): 
            ULRSSframe = solution.SSs[-1]
            #solution.write_as_pdb("part."+solution.tag+".pdb")

            # Interface to Rosetta Jump!
            threads = [sol[2] for sol in solution.threads] #sol = (begin,end,cenpos)

            jump_at_ULR.append_moveset_from_match(pose_ext,solution,threads,iSS=-1)

            '''
            #oldway
            newjumps,stubs,ijump = rosetta_utils.SS_to_jump( pose_ext,
                                                             ULRSSframe, #frame info from solution
                                                             FTInfo,
                                                             threads )

            # validation as pose
            for i,newjump in enumerate(newjumps):
                
                
                for ires in range(ULRSSframe.nres):
                    resno = ULRSSframe.begin + ires + shift
                    phi,psi,omg = ULRSSframe.tors[ires]
                    if abs(phi) < 360.0: pose.set_phi(resno,phi)
                    if abs(psi) < 360.0: pose.set_psi(resno,psi)
                    if abs(omg) < 360.0: pose.set_omega(resno,omg)
                pose.dump_pdb("full.%s.t%03d.pdb"%(solution.tag,cenres))
            '''
            
    #validation -- new way
    print("\n[ChunkSampler/main] ================== Scan finished ====================")
    print(" - validation -- writing pdbs")
    jump_at_ULR = FTInfo.get_jump(ulrSSs[0].reslist)
    #jump_at_ULR = FTInfo.jumps[ulrSSs[0].jumpid] #bug?
    poses_move = jump_at_ULR.moveset_to_pose( pose_ext, mode='all', report_pdb=True )

    return pose_ext, FTInfo #are these sufficient??
    
if __name__ == "__main__":
    opt = arg_parser(sys.argv[1:])
    PR.init('-mute all')
    main(opt)
