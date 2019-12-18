import sys,os,glob
import numpy as np
import pyrosetta as PR
import multiprocessing
initcmd = '-hb_cen_soft -overwrite -mute all'
PR.init(initcmd)

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SCRIPTDIR+'/PseudoCritics')
import Error2cst, ErrorPredictor

sys.path.insert(0,SCRIPTDIR+'/PseudoActors')
import ChunkFilter,SSpredictor,FragPicker

sys.path.insert(0,SCRIPTDIR+'/Rosetta')
import miniRosettaFold

# ignore all tensorflow warnings -- why is this not working???
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

ERRORBINS = np.array([-99.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.5,0,0.5,1.5,3.0,7.0,12.5,17.5,99.0])
VALLHOME= '/home/minkbaek/DeepLearn/torsion/vall'

class LocalOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def arg_parser(argv):
    import argparse
    opt = argparse.ArgumentParser()
    ## Input
    opt.add_argument('-initpdb', dest='initpdb', required=True, help='Input pdb file')
    opt.add_argument('-a3m', dest='a3m', required=True, help='Input a3m file')
    opt.add_argument('-ulr', dest='ulr_s', nargs='+', required=True, \
                     help='ULRs should be sampled. (e.g. 5-10 16-20). If not provided, whole structure will be sampled')
    opt.add_argument('-p', dest='ncores', default=20, help="N CPU processesors ")
    opt.add_argument('-native', dest='native', help='Input ref file')
    opt.add_argument('-predict_ulr', dest='predict_ulr', default=False, action='store_true',
                     help='predict ulr instead of providing')
    opt.add_argument('-use_chunklib', dest='use_chunklib', default=False, action='store_true',
                     help='')
    opt.add_argument('-nstruct', dest='nstruct_per_iter', type=int, default=20, help="N structures to sampler every iter")

    opt.ulrs_static = None
    params = opt.parse_args()
    if params.ulr_s != []:
        params.ulrs_static = [range(int(word.split('-')[0]),int(word.split('-')[1])) for word in params.ulr_s]
    return params

def collect_E_ulrs_from_SSpred(ulrs,SS3pred_np):
    sortable = [(len(ulr),ulr) for ulr in ulrs]
    sortable.sort()
    sortable.reverse()

    # Find longest ULR having at least 3 res as strand
    ulr_major = None
    for n,ulr in sortable:
        nE = 0
        for res in ulr:
            if SS3pred_np[res-1] == 'E': nE +=1
        #print( ulr, len(ulr), nE )
        if nE >= 3:
            ulr_major = ulr
            break
    return ulr_major

class OneIterator:
    def __init__(self,pdb):
        self.refpdb = pdb
        self.SS3pred = False
    
    def PickFragment(self,a3m):
        opt = LocalOptions( pdb_fn=self.refpdb, a3m_fn=a3m,
                            n_layer=20, n_1d_layer=12,
                            n_feat=64, n_bottle=32, dilation=[1,2,4,8],
                            model_dir='%s/models/SSpred'%(SCRIPTDIR),
                            report_files=False)
            
        # 1. Mink's prediction & fragment picking
        sspred = SSpredictor.Predictor() #Get an instance
        SS_prob, SS3_prob, tors_prob, _ = sspred.run(opt) #numpy objects
        self.SS3pred = SS3_prob #store for future usage

        # options for picker
        opt_pick = LocalOptions( title="mink", a3m_fn=a3m,
                                 n_frag=25, batch_size=64,
                                 vall_fn='%s/data/vall.jul19.2011.vonMises.npy'%VALLHOME,
                                 vall_full='%s/data/vall.jul19.2011.json'%VALLHOME
                                 )

        # directly pass numpy objects generated from SSpredictor
        FragPicker.main( params=opt_pick,
                         pred_tor_np=tors_prob, pred_SS_np=SS_prob,
                         n_mers=[3,9],
                         title='mink')

        # expected output:
        fraglib_big = 'mink.9mers'
        fraglib_small = 'mink.3mers'
        
        return fraglib_big, fraglib_small

    def ErrorPrediction(self,workpath='tmp',ulrs=[],
                        generate_cst=True,
                        predict_ulr=False, ncore=1 ):
        
        # make sure there is any pdb inside workpath
        pdbs = glob.glob(workpath+'/*pdb')
        if len(pdbs) == 0:
            sys.exit('No pdb found in work directory: %s!'%workpath)
        opt_pred = LocalOptions( infolder=workpath, outfolder=workpath,
                                 process=ncore, multiDecoy=False,
                                 noEnsemble=True, leavetemp=False,
                                 verbose=False )
        
        lddts_pred = ErrorPredictor.main(opt_pred)
        Qs = {}
        for key in lddts_pred:
            Qs[key] = np.mean(lddts_pred[key])
        
        if generate_cst:
            #csts will be stored as '[prefix].cst' & '[prefix].fa.cst' for cen & fa, respectively
            for i,pdb in enumerate(pdbs):
                prefix = pdb[:-4]
                errorpred_np = prefix+'.npz'
                opt_cst = LocalOptions( npz=errorpred_np, pdb=pdb, prefix=prefix,
                                        ulrs_static=ulrs,
                                        ulr_pred=predict_ulr, softcst='none',
                                        crdcst=False )
            
                cstgenerator = Error2cst.Error2cst(opt_cst)
                cstgenerator.run(ebin_defs=ERRORBINS) #hard-coded

                # no logic yet to update ulrs_pred from multi models-- hack for now
                if i == 0:
                    self.ulrs_pred = cstgenerator.ulrs_dynamic #store it anyways
        return Qs

    def ChunkLibGeneration(self,ulrs,libsize=10):
        chunk_filter = ChunkFilter(self.refpdb)
        print( "Searching for list of ULRs to try chunk insertion..." )
        ulr_major = chunk_filter.collect_E_ulrs_from_SSpred(ulrs,self.SS3pred)
        
        if ulr_major == None:
            print( "No possible chunk found... skipping chunk lib gen" )
            return False
        else:
            chunk_filter.read_SSpred(self.SS3pred)
            chunk_filter.read_estogram(self.errornpy,ebin_defs=ERORRBINS)

        opt = LocalOptions( nmax=libsize, ulrs=ulr_major, pdb=self.refpdb,
                            SStype='EE' ) 
        stat = chunk_filter.run(opt)
        return stat

#tmp: hacky interface version
def run_minirosetta(args):
    refpdb,ulrs,nstruct,fragbig,fragsmall,outprefix = args

    #opt = LocalOptions( pdb_fn=refpdb, ulr=ulrs, nstruct=nstruct,
    #                    frag_fn_big=fragbig, frag_fn_small=fragsmall,
    #                    cen_only=True, prefix=outprefix )
    # let's use local argparser because there are too many args...
    argv = ['-s',refpdb,
            '-nstruct',str(nstruct),
            '-frag_fn_big',fragbig,
            '-frag_fn_small',fragsmall,
            '-cen_only',#Centroid modeling only!
            '-prefix',outprefix,
            '-mute'] #mute!

    opt = miniRosettaFold.arg_parser(argv)
    opt.ulr_s = ulrs

    runner = miniRosettaFold.Runner(opt) # interface through pdb files?
    runner.apply() 
    #Caution: dies if passes pyrosetta.pose
    #expected output: outprefix+"_0000.pdb"

def launch_Rosetta_jobs(initpdb,ulrs,fraglib_big,fraglib_small,nstruct_total,ncore=20,
                        nselect=50,pdbpath='./'):
    if nstruct_total < ncore:
        n_per_core = 1
        ncore = nstruct_total
    else:
        n_per_core = int(nstruct_total/ncore)
    args = []
    for i in range(ncore):
        args.append((initpdb,ulrs,n_per_core,fraglib_big,fraglib_small,'%s/gen%d'%(pdbpath,i)))

    print( "Launch %d jobs on %d cores..."%(len(args),ncore))
    launcher = multiprocessing.Pool(processes=ncore)

    ###
    # If multiprocessing worked with Pyrosetta:
    #ans = launcher.map(run_minirosetta, args)
    #poses = []
    #for an in ans: poses += an
    #+Some selection should happen here
    #return poses[:nselect]
    
    # interface through pdb files instead...
    launcher.map( run_minirosetta, args ) # run_minirosetta is a pyrosetta module
    pdbs = glob.glob('%s/gen*pdb'%(pdbpath))
    
    #+some selection should happen here
    
    return pdbs[:nselect]
    
def test_job():
    opt = arg_parser(sys.argv[1:])
    
    ulrs = opt.ulrs_static # requires initial ulr definition!
    it = OneIterator(opt.initpdb)
    
    ##Actor stage.
    # Mink's SSpred & PickFragment
    # fraglib_big,small are "strings"
    fraglib_big,fraglib_small = it.PickFragment(opt.a3m) #output are file names
    if not os.path.exists(fraglib_big) or not os.path.exists(fraglib_small):
        sys.exit("No fraglib")

    # (optional). Chunk library generation -- not debugged yet
    if opt.use_chunklib:
        stat = it.ChunkLibGeneration(ulrs)
        if stat: chunk_filter.report('chunks.extra.txt') #Just reporting
        else: print( "Skipping chunk insertion" )

    # Transition stage -- Rosetta modeling
    # store pdb at pwd+pdbpath
    # currently interfacing through pdb files because of pyrosetta+multiprocessing issue...
    launch_Rosetta_jobs(initpdb = opt.initpdb,
                        ulrs = ulrs, 
                        fraglib_big = fraglib_big, #update
                        fraglib_small = fraglib_small, #update
                        nstruct_total = opt.nstruct_per_iter,
                        ncore = opt.ncores,
                        pdbpath = 'tmp' ) #dump pdbs to pdbpath

    ##Critic stage.
    # get model qualities
    Qs = it.ErrorPrediction( workpath='tmp', # read all pdbs from workpath
                             ulrs = ulrs,
                             generate_cst=True,
                             ncore=opt.ncores ) #just for cst generation fxnality check

    print("Generated model qualities:",Qs)

    # (optional). ULR re-declaration
    if opt.predict_ulr: #if no ULR defined in input, bring from predicted
        ulrs = it.ulrs_pred

if __name__ == "__main__":
    test_job()
