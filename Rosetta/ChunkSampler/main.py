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
    parser.add_argument('-SSpred', dest='SSpred', required=True,
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

    # Read in ULR/Jump info from estogram2FT
    # estogram should be always supported
    FTInfo = estogram2FT.main(pose,opt) #== FoldTreeInfo class

    # setup pose info
    # poseinfo.extraSS << ULR SSs
    poseinfo = PoseInfo(pose,opt,FTInfo,SSpred=opt.SSpred)

    # read static TERM database
    # todo -- make loop db in same format
    full_db = TERMDB(opt)
    
    # get list of potential ULR anchors
    anctypes = [SS[:-1] for SS in opt.config['SSTYPE_SUPPORTED']]
    SS1anchors, SS2anchors = poseinfo.get_ULR_anchors(FTInfo,anctypes)

    # scan through SScombs 
    for SScomb in full_db:
        # retrieve db for certain SS combinations
        db = full_db[SScomb]

        # setup matcher
        matcher = Matcher( db, debug=opt.report_pdb)  

        # scan through ulrs
        for ulr_t in poseinfo.ULRs: #ulr_t: in SSclass form
            matcher.place_ULR_at_anchors( poseinfo, SS1anchors, ulr_t )
            matcher.place_ULR_at_anchors( poseinfo, SS2anchors, ulr_t )
            
        matcher.do_filter( poseinfo, FTInfo.ulrs,
                           nmax = opt.config['NFILTER'] ) # prv. Rosetta app
        matcher.report_as_npz('%s.%s.chunk.npz'%(opt.prefix,SScomb))

def test(opt):        
    pose = PR.pose_from_pdb(opt.pdb)
    
    # manually assign ulr: this will make estogram2FT skip ulr search
    #opt.ulrs = [list(range(22,36))]
    
    # Read in ULR/Jump info from estogram2FT
    # estogram should be always supported
    FTInfo = estogram2FT.main(pose,opt) #== FoldTreeInfo class

    # setup pose info
    poseinfo = PoseInfo(pose,opt,FTInfo,SSpred=opt.SSpred)

    # read static TERM database
    # todo -- make loop db in same format
    full_db = {}
    for SStype in opt.config['SSTYPE_SUPPORTED']:
        full_db[SStype] = TERMDB(opt,SStype)
    
    # get list of potential ULR anchors
    anctypes = [SS[:-1] for SS in opt.config['SSTYPE_SUPPORTED']]
    SS1anchors, SS2anchors = poseinfo.get_ULR_anchors(FTInfo,anctypes,
                                                      report=opt.debug)

    # Defined as list of SSclass that are from ULR
    ulrSSs = FTInfo.extraSS_at_ulr

    # scan through SScombs 
    for SScomb in ['HH']:#full_db:
        # retrieve db for certain SS combinations
        db = full_db[SScomb] #TERMDB class
        print( "\n"+"="*100+"\nSEARCHING through TERM db %s with %d candidates"%(SScomb,len(db.db)))

        # setup matcher
        matcher = Matcher( db, opt, prefix=SScomb, debug=opt.write_pdb )

        # Matcher.MatchSolutions containing .SSs (and .SS.bbcrds)
        solutions = []
        for ulrSS in ulrSSs:
            solutions += matcher.place_ULR_at_anchors( poseinfo, SS1anchors[:2], ulrSS )
            solutions += matcher.place_ULR_at_anchors( poseinfo, SS2anchors[:2], ulrSS )
        print( len(solutions) )

        # prv. Rosetta app
        #matcher.do_filter( poseinfo, ulrjumps,
        #                   nmax = opt.config['NFILTER'] ) 
        
        for i,solution in enumerate(solutions):
            ULRanc = solution.anchorres[-1]
            # Rosetta Jump!
            newjump = rosetta_utils.SS_to_jump( pose,
                                                solution.SSs_term[-1], #SSclass
                                                ULRanc, #integer
                                                FTInfo,
                                                change_pose=True)
            pose.dump_pdb(solution.tag+".pdb")

        #matcher.report_as_npz('%s.%s.chunk.npz'%(opt.prefix,SScomb))
        
if __name__ == "__main__":
    opt = arg_parser(sys.argv[1:])
    PR.init('-mute all')
    #main(opt)
    test(opt)
