import sys,os
import numpy as np
import multiprocessing as mp
from pyrosetta import *
MYPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,'%s/../Rosetta/basic'%MYPATH)
import rosetta_utils
import time
from numba import cuda

DLPATH='/projects/casp/RefinementScripts/DeepAccNetMSA/latest'
#ROSETTAPATH=os.environ.get("ROSETTAPATH")

def relax(args):
    pdb,outpdb,relaxscript = args
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    sf_fa = create_score_function('ref2015_cart')
    relax = rosetta.protocols.relax.FastRelax(sf_fa,relaxscript)
    relax.set_movemap(mmap)

    pose = pose_from_file(pdb)

    relax.apply(pose)
    pose.dump_pdb(outpdb)

def relax_multi(ncore,args):
    pdbs_out = [arg[1] for arg in args]
    
    while True:
        if len(args) > 0:
            inpdb,outpdb,relaxscript = args.pop()
            if os.path.exists(outpdb): continue
            os.system("%s/relax_w_script.sh %s %s %s &"%(MYPATH,inpdb,outpdb,relaxscript))

        ndone = len([pdb for pdb in pdbs_out if os.path.exists(pdb)])
        if ndone == len(pdbs_out):
            break
        time.sleep(1)
        
class FaQRunner:
    def __init__(self,relax=True,
                 relaxscript="cart1.script",
                 verbose=False
                 ):
        self.relax = relax
        self.relaxscript = relaxscript
        self.verbose = verbose
        self.poses = []
        self.scores = []

    def apply(self,poses,msanpz,tmppath="tmp",ncore=10):
        ## Force empty GPU memory allocated by others
        # This is a very stupid way but only way that works currently...
        # should be replaced by a safer way later
        device = cuda.get_current_device()
        device.reset()
        
        pwd = os.getcwd()
        n = len(poses)
        ncore = min(n,ncore)
        if not os.path.exists(tmppath):
            os.mkdir(tmppath)
        
        # Hack to run relax in mp
        # key is not-to-interface-through-pyrosettaClass (e.g. pose)
        os.chdir(tmppath)
        if self.relax:
            print(" - Relax %d structs in %d cores at temporary directory %s..."%(len(poses),ncore,tmppath))
            args = []
            for i,pose in enumerate(poses):
                inpdb  = "in.%03d.pdb"%i
                outpdb = "out.%03d.pdb"%i
                pose.dump_pdb(inpdb)
                args.append((inpdb,outpdb,self.relaxscript))
                
            # 1. through multiprocessing
            #a = mp.Pool(processes=ncore)
            #a.map(relax,args)
            # 2. or call through shell instead if there is any memory issue
            relax_multi(ncore,args)
            
            # clean inputs
            os.system('rm in.*pdb')
        else:
            for i,pose in poses:
                outpdb = "out.%03d.pdb"%i
                pose.dump_pdb(outpdb)

        # run FaQscorer
        if self.verbose:
            print('CMD: python %s/scripts/ErrorPredictorMSA.py -p %d %s ./ >& DANmsa.logerr'%(DLPATH,ncore,msanpz))
        os.system('python %s/scripts/ErrorPredictorMSA.py -p %d %s ./ >& DANmsa.logerr'%(DLPATH,ncore,msanpz))

        npzs = ['out.%03d.npz'%i for i in range(n) if os.path.exists('out.%03d.npz'%i)]
        if len(npzs) != n:
            sys.exit("ERROR: DAN-fa outputs not consistent with input, die!")

        self.scores = np.zeros(n)
        self.poses = []
        for i,npz in enumerate(npzs):
            Q = np.mean(np.load(npz)['lddt'])
            pose = pose_from_file(npz.replace('.npz','.pdb'))

            self.scores[i] = Q
            self.poses.append(pose)
            if self.verbose:
                print("- report FaQscore: %-20s %6.4f"%(npz[:-4],self.scores[i]))

        os.chdir(pwd)
        os.system('rm -rf %s'%tmppath)

    def get_best(self):
        if len(self.poses) == 0 or len(self.scores) == 0:
            return None,0.0
        ibest = np.argmax(self.scores)
        return self.poses[ibest], max(self.scores)

    def dump_silent(self,outf,refpose=None,outprefix="sample"):
        for i,pose in enumerate(self.poses):
            rosetta_utils.report_pose(pose,
                                      tag="%s.%02d"%(outprefix,i),
                                      extra_score = [("Qfa",self.scores[i])],
                                      outsilent=outf,
                                      refpose=refpose)
        

if __name__ == "__main__":
    init('-mute all')
    pdbs = [l[:-1] for l in open(sys.argv[1])]
    msanpz = sys.argv[2]
    
    poses = [pose_from_file(pdb) for pdb in pdbs]
    AAscorer = FaQRunner(verbose=True)
    AAscorer.apply(poses,'msa.npz')

    pose_best,score_best = AAscorer.get_best()
    print("Score: %.3f"%score_best)
    pose_best.dump_pdb("best.pdb")
    
