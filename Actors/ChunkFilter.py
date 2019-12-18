import sys,copy,glob,os
import numpy as np
#MYFILE = os.path.abspath(__file__)
#SCRIPTDIR = MYFILE.replace(MYFILE.split('/')[-1],'')
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/ChunkSampler'%SCRIPTDIR)
from Protein import *
from Jump import Jump, JumpDB

def arg_parser(argv):
    opt = argparse.ArgumentParser()

    ## Input
    opt.add_argument('-pdb', dest='pdb', required=True, help='Input pdb file')
    opt.add_argument('-ulr', dest='ulrstr', nargs='+', required=True, \
                     help='ULRs should be sampled. (e.g. 5-10 16-20). If not provided, whole structure will be sampled')
    opt.add_argument('-SStype', dest='SStype', required=True, help='SS type of chunk')
    opt.add_argument('-nmax', dest='n', default=10, help="Library size filtered")
    opt.add_argument('-cst', dest='cst', help='Input cst file')
    opt.add_argument('-offset', dest='offset', default=0.0, help="Offset in filtering score")
    opt.add_argument('-write_pdb', dest='write_pdb', default=False, action="store_true",
                     help='report library as pdb files')

    if len(argv) == 1:
        opt.print_help()
        return

    params = opt.parse_args()
    params.ulrs = []
    for word in params.ulrstr.split(','):
        res1,res2 = word.split('-')
        params.ulrs.append(range(int(res1),int(res2)+1))

    if params.SStype not in ['EE','HHH','HH','EEH','EEE']:
        sys.exit("No SStype supported: %s (lexical 'EE','HHH','HH','EEH')"%params.SStype)
    
    return params

def lookup(vals,d,dbins):
    ibin = min(max(0,int(d)-3),len(vals)-1)
    for dbin in dbins:
        if ibin < dbin:
            return vals[dbin]
    return 0.0

def pdb2crd(pdb):
    crds = {}
    for l in open(pdb):
        if not l.startswith('ATOM'): continue
        atm = l[12:16].strip()
        aa  = l[16:20].strip()
        resno = int(l[22:26].strip())
        if aa == 'GLY' and atm == 'CA':
            crds[resno] = (float(l[30:38]),float(l[38:46]),float(l[46:54]))
        elif atm == 'CB':
            crds[resno] = (float(l[30:38]),float(l[38:46]),float(l[46:54]))
    return crds

class ChunkFilter:
    def __init__(self):
        self.SS3prob = None
        self.estogram = None #as npy
        self.dbins = None
        self.dbpath = '%s/TERMlib/'%SCRIPTDIR
        self.ulrs = None

    def run(self,params=None):
        if params == None:
            self.opt = arg_parser(sys.argv[1:])
        else:
            self.opt = params

        if self.opt.ulrs == [] or self.opt.ulrs == None:
            sys.exit('No ulr defined!')
            
        protein = Protein(self.opt.pdb,self.opt.ulrs,permissive_SSdef=True)
        
        SStypes = [self.opt.SStype] #support only one type for now...
        maxlibrank = 100000
        
        if SStype in ['EE','EEE']:
            SStypes.append('input')
            maxlibrank = {'EE':100,'EEE':10000}
        else:
            if SStype == 'HH':
                maxlibrank = 10000
            protein.permissive_SSdef = False

        scannedDB = self.read_DB(protein,SStypes,maxlibrank=maxlibrank)

        # replaces rosetta app -- still interface through matchf
        self.select_matches(self.opt.pdb,scannedDB)
        
        return (len(self.selected) >= self.opt.n) #"stat"

    def read_DB(self,combinations, #=['EE','EEE','HEE','HHH'],
                maxlibrank=10000):

        DB = JumpDB(self.dbpath,maxlibrank=maxlibrank)
        DB.set_report_pdb(self.opt.write_pdb)

        ## not generalized to have more than 1 DB... is it necessary?
        if 'EE' in combinations:
            #print '=== Searching DBEE ==='
            scannedDB = DB.dbtype["EE"]
            scannedDB.scan_through_unpaired_strands(protein,nanchor=1)
            if 'input' in combinations:
                #print "=== Searching input structure for EEtype ==="
                protein.make_threads_from_ulr('EE',DB_EE.out,
                                              DB_EE.report_pdb)

        if 'HH' in combinations:
            #print '=== Searching DBHH ==='
            scannedDB = DB.dbtype["HH"]
            scannedDB.duplication_rmsd_cut = 8.0
            scannedDB.scan_through_exposed_helix(protein)
        
        if 'EEE' in combinations:
            #print '=== Searching DBEEE ==='
            scannedDB = DB.dbtype["EEE"]
            scannedDB.scan_through_unpaired_strands(protein,nanchor=2)

        if 'HEE' in combinations:
            #print '=== Searching DBHEE ==='
            scannedDB = DB.dbtype["HEE"]
            scannedDB.duplication_rmsd_cut = 8.0 
            scannedDB.scan_through_exposed_SSpair(protein,nanchor=2,SStype='E')

        if 'HHH' in combinations:
            #print '=== Searching DBHHH ==='
            scannedDB = DB.dbtype["HHH"]
            scannedDB.duplication_rmsd_cut = 3.0  
            scannedDB.scan_through_exposed_SSpair(protein,nanchor=2,SStype='H')
        #if 'EET' in combinations:
        #    DB_EET = DB.dbtype["EET"]
        #    DB_EET.out = file("EET.match",'w')
        #    DB_EET.scan_through_exposed_SSpair(protein,nanchor=2,SStype='E')

        return scannedDB ## JumpDBtype
            
    def read_SSpred(self,SS3prob_npy):
        self.SS3prob = np.load(SS3prob_npy)

    def read_estogram(self,pdb,estogram_npy,ebin_defs,
                      reference_npy=None):
        pdbCBcrds = pdb2crd(pdb,'CB')
        #ebin_defs: how input distogram defined
        # e.g.: np.array([-99.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.5,0,0.5,1.5,3.0,7.0,12.5,17.5,99.0])

        self.estogram = np.load(estogram_npy)
        nres = len(self.estogram)

        if reference_npy != None:
            refgram = np.load(reference_npy)
            self.estogram /= refgram #Check

        # eval on d0 < 40 Ang list
        in_contacts = [[False for i in range(nres)] for j in range(nres)]
        self.dbins = np.zeros((nres,nres,15))
        for ires in range(1,nres):
            for jres in range(ires+1,nres+1):
                d = distance(pdbCBcrds[ires],pdbCBcrds[jres])
                self.dbins[ires][jres] = self.dbins[jres][ires] = ebin_defs+d

    def select_matches(self,scannedDB):
        pdbCBcrds = pdb2crd(self.opt.pdb,'CB')
        crds_noulr = {}
        ulrres = []
        for ulr in self.opt.ulrs: ulrres += ulr
        for res in pdbCBcrds:
            if res not in ulrres: crds_noulr[res] = pdbCBcrds[res]
        
        threadble = []
        for match in scannedDB.chunkmatches: #retrieve after JumpDBtype.scan_region function
            threads = match.threads
            for i,thread_res in enumerate(threads):
                score_dist = thread_and_score( crds_noulr, match.crds, thread_res )
                score_SS = SSmatch_score( thread_res, match.SStype )
                score = -(score_dist + score_SS) #make it negative
                threadable.append((score,match,thread_res[0],thread_res[-1]))

        threadable.sort()
        covered = {}
        self.selected = []
        while npick < self.opt.n:
            (score,match,begin,end) = threadable[0]
            self.selected.append((match,begin,end))
            covered[begin] += 1
            #print("pick: %f %s"%(begin, conts[0][2]))

            # add penalty to covered/picked
            for j,thread in enumerate(threadable):
                begin = thread[2]
                if j == 0:
                    thredable[j][0] += 9999 #score
                else:
                    threadable[j][0] += covered[thread]*5.0
            threadable.sort() #resort reflecting penalty

    def thread_and_score(self,crds_noulr,crds_insertd,thread_res):
        if self.estogram == None:
            sys.exit('No estogram read in!')
            
        score = 0.0
        for i,res1 in enumerate(thread_res):
            # score thread res
            for res2 in crds_noulr:
                d = distance(crds_insertd[res1],crds_noulr[res2])
                score += lookup(self.estogram[res1][res2],self.dbins,d)
        return score

    def SSmatch_score(self,thread_res,SStype):
        if self.SS3prob == 'None':
            sys.exit('No SS3 probability read in!')
            
        score = 0.0
        for res in thread_res:
            score += self.SS3prob[res]
        return score/len(thread_res) #average probability

    def report(self,outf):
        # write minimal info to interface with pyhyb txt parser
        out = open(outf,'w')
        for i,(match,begin,end) in enumerate(selected):
            out.write("CHUNK %04d\n"%i)
            out.write("THREAD %3d %3d\n"%(begin,end))
            for crd in match.bbcrd:
                crdl = 'COORD'
                for atm in ['N','CA','C','O']:
                    crdl += ' %8.3f %8.3f %8.3f'%(crd[atm])
                out.write(crdl)
            out.write("END\n")
        out.close()
        
if __name__ == "__main__":
    a = ChunkFilter()
    a.run()
