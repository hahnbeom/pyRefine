import sys,os
import copy
import numpy as np
import scipy as sp
import pyrosetta as PR
from FoldTreeInfo import FoldTreeInfo
import types

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0,SCRIPTDIR+'/../utils/')
import myutils

######################################
def arg_parser(argv):
    #####################
    # Parsing arguments
    #####################
    import argparse
    
    parser = argparse.ArgumentParser(
        description="A script to subdivide estograms via the charikar algorithm [2002]",
        epilog="v0.0.1")
    
    parser.add_argument("npz",
                        action="store",
                        help="input prediction .npz file")

    parser.add_argument("pdb",
                        help="input pdb file corresponding to npz file")
    
    parser.add_argument("--outfile",
                        action="store",
                        default=None,
                        help="output .csv file")

    ## options for SubGraph
    parser.add_argument("--subdef_conserve",
                        "-c", action="store",
                        type=float,
                        default=1.0,
                        help="threshold for errors [0.5, 1, 2, 4], default:1")
    
    parser.add_argument("--subdef_confidence",
                        "-conf", action="store",
                        type=float,
                        default=0.5,
                        help="threshold for confidence [0~1], default:0.5")

    '''parser.add_argument("--min_coverage",
                        action="store",
                        type=float,
                        default=0.0, #by default always converge
                        help="Dynamically assign until this coverage")'''
    
    parser.add_argument("--subdef_density",
                        "-d", action="store",
                        type=float,
                        default=0.9,
                        help="threshold for subgraph density [0~1], default:0.9")
    
    parser.add_argument("--algorithm",
                        "-a", action="store",
                        type=int,
                        default=0,
                        help="algorithm choice edge based/node based density, default (recommended): 0")

    ## options for ULR
    parser.add_argument("--ulrmode",
                        default="conservative",
                        help="[dynamic/conservative]")
    parser.add_argument("--ulr_fmin",type=float,default=0.15,
                        help="minimum fraction of residues assigned as ULR, default:0.9")
    parser.add_argument("--ulr_fmax",type=float,default=0.25,
                        help="minimum fraction of residues assigned as ULR, default:0.9")


    if len(argv) < 1:
        parser.print_help()
        sys.exit(1)
        
    opt = parser.parse_args(argv)
    return opt

def add_default_options_if_missing(opt=types.SimpleNamespace()):
    #opt = 
    if not hasattr(opt,"outfile"): opt.outfile = None
    if not hasattr(opt,'debug'): opt.debug = False

    # Sub
    if not hasattr(opt,"subdef_conserve"): opt.subdef_conserve = 1.0
    if not hasattr(opt,"subdef_confidence"): opt.subdef_confidence = 0.5
    if not hasattr(opt,"min_coverage"): opt.min_coverage = 0.0
    if not hasattr(opt,"density"): opt.density = 0
    if not hasattr(opt,"algorithm"): opt.algorithm = 0
    if not hasattr(opt,"subs"): opt.subs = [] #placeholder
    
    # ULR
    if not hasattr(opt,"ulrmode"): opt.ulrmode = "dynamic"
    if not hasattr(opt,"ulrs"): opt.ulrs = [] #placeholder
    if not hasattr(opt,"ulr_fmin"): opt.ulr_fmin = 0.15
    if not hasattr(opt,"ulr_fmax"): opt.ulr_fmax = 0.25
    #return opt
    
def Qres2dev(Qres):
    # function converting lddtscale to CA/CB dev
    # collected from statistics
    if Qres > 0.7:
        return 1.0 #minimum val
    elif Qres < 0.2:
        return 100.0 #maximum value
    else:
        return np.exp(5*(0.7-Qres)) #range from 1 to 20
    
######################################
# ULR
def estogram2ulr(estogram,opt):
    nres = len(estogram) #pred is lddt-per-res

    # 1. Make soft & normalized Qres
    # non-local distance accuracy -- removes bias from helices
    Qres = np.zeros(nres)
    for i in range(nres):
        n = 0
        for j in range(nres):
            if abs(i-j) < 13: continue ## up to 3 H turns
            n += 1
            Qres[i] += sum(estogram[i][j][6:9]) #+-1
        Qres[i] /= n
    Qlr = np.mean(Qres)
    
    Qres_soft = np.zeros(nres) #soften by 9-window sliding
    for i in range(nres):
        n = 0
        for k in range(-4,5):
            if i+k < 0 or i+k >= nres: continue
            n += 1
            Qres_soft[i] += Qres[i+k]
        Qres_soft[i] /= n

    # 2. return either binary ULR pred or max-possible-deviation
    if opt.ulrmode == 'dynamic':
        ulrres = pred_ULR(Qlr,Qres_soft,dynamic=True)
    else:
        ulrres = pred_ULR(Qlr,Qres_soft,
                        fmin=opt.ulr_fmin,
                        fmax=opt.ulr_fmax)

    ##split to regions
    # currently only two options...
    ulrs = myutils.list2part(ulrres)
    
    # translate to max dev in Angstrom
    return ulrs, Qres_soft

def pred_ULR(Q,Qres_soft,fmin=0.15,fmax=0.25,dynamic=False):        
    if dynamic: #make it aggressive!
        fmax = 0.3+0.2*(0.55-Q)/0.3 #Qlr range b/w 0.25~0.55
        if fmax > 0.5: fmax = 0.5
        if fmax < 0.3: fmax = 0.3
        fmin = fmax-0.1
        print( "dynamic ULR: lddtPred/fmin/fmax: %8.5f %6.4f %6.4f"%(Q, fmin, fmax))
        
    QresCUT = 0.3 #initial
    for it in range(50):
        factor = 1.1
        if it > 10: factor = 1.05
        
        is_ULR = [False for ires in Qres_soft]
        for i,Qres in enumerate(Qres_soft):
            if Qres < QresCUT: is_ULR[i] = True
        myutils.sandwich(is_ULR,super_val=True,infer_val=False)
        myutils.sandwich(is_ULR,super_val=False,infer_val=True)
        f = is_ULR.count(True)*1.0/len(is_ULR)

        if f < fmin: QresCUT *= factor
        elif f > fmax: QresCUT /= factor
        else: break

    ULR = []
    for i,val in enumerate(is_ULR):
        if val: ULR.append(i)
    return myutils.trim_lessthan_3(ULR,len(Qres_soft)-1)
    
######################################
## SUB
# Estogram binarization
def binrize_estogram(estogram, exclude=[], threshold = 1):
    assert threshold in [0.5, 1, 2, 4]
    if threshold == 0.5:
        inds = (7,8)
    elif threshold == 1:
        inds = (6,9)
    elif threshold == 2:
        inds = (5,10)
    elif threshold == 4:
        inds = (6,11)
    output = np.sum(estogram[:,:,inds[0]:inds[1]], axis=2)

    #clear excluded res
    for i in exclude:
        output[:,i] = output[i,:] = 0.0
    return output

def chunk2SS(chunk,SS3):
    # extend through the same SS
    i,e = (chunk[0],chunk[-1])
    SSi,SSe = (SS3[i],SS3[e])
    SSchunk = copy.deepcopy(chunk)
    while True: #N-term: back-trace
        i -= 1
        if i < 0 or SS3[i] == 'L': break
        if SS3[i] == SSi: SSchunk.append(i)
        else: break

    while True: #N-term: back-trace
        e += 1
        if e >= len(SS3) or SS3[e] == 'L': break
        if SS3[e] == SSe: SSchunk.append(e)
        else: break
        
    SSchunk.sort()
    return SSchunk
        

# Edge percentage (Basically looks for cliques)
def density(graph):
    base = np.prod(graph.shape) - graph.shape[0]
    if base==0: return 0
    return np.sum(graph*(1-np.eye(graph.shape[0])))/base

# Density (more formal definition)
def density2(graph):
    return np.sum(graph*(1-np.eye(graph.shape[0])))/graph.shape[0]

def charikar(g, vs=None):
    if vs is None:
        vs = np.arange(g.shape[0])
    cur_graph = g
    cur_vs = vs
    data = [(cur_graph, cur_vs, density(cur_graph))]
    while cur_graph.shape[0] != 1:
        # Calculate degrees and sort
        cur_degs = np.sum(cur_graph*(1-np.eye(cur_graph.shape[0])), axis=0)
        temp = [(cur_degs[i], cur_vs[i]) for i in range(cur_graph.shape[0])]
        temp.sort()

        # Get everytthing but least connected v
        least_v = temp[0][1]
        idxs = np.where(cur_vs!=least_v)[0]

        # Remove least connectedv
        cur_graph = cur_graph[idxs, :][:, idxs]
        cur_vs = cur_vs[idxs]

        # Calcuate density
        data.append((cur_graph, cur_vs, density(cur_graph)))
    data.sort(key=lambda x:x[2], reverse=True)
    return data[0]

def find_blocks(g, vs=None, visualize=False):
    if vs is None:
        vs = np.arange(g.shape[0])
    cur_g = g
    cur_vs = vs
    outputs = []
    while cur_g.shape != (0,0):
        #print(d, cur_g.shape, cur_vs)
        subgraph, subgraph_vs, d = charikar(cur_g, cur_vs)
        keep_vs = np.array([v for v in cur_vs if not v in subgraph_vs])
        keep_inds = np.array([i for i in range(len(cur_vs)) if cur_vs[i] in keep_vs])
        remove_inds = np.array([i for i in range(len(cur_vs)) if not cur_vs[i] in keep_vs])
        
        if visualize:
            plt.figure()
            plt.subplot(131)
            plt.imshow(cur_g, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            t = np.zeros(cur_g.shape)
            for i in remove_inds:
                for j in remove_inds:
                    t[i,j] = 1
            plt.subplot(132)
            plt.imshow(t, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            plt.subplot(133)
            plt.imshow(subgraph, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            plt.show()

        if d==0:
            for i in subgraph_vs:
                outputs.append(np.array([i]))
            break
        cur_g = cur_g[keep_inds, :][:, keep_inds]
        cur_vs = keep_vs
        outputs.append(subgraph_vs)
    return outputs

def charikar2(g, vs=None, th=0.8):
    if vs is None:
        vs = np.arange(g.shape[0])
    cur_graph = g
    cur_vs = vs
    data = [(cur_graph, cur_vs, density2(cur_graph))]
    while cur_graph.shape[0] != 1:
        # Calculate degrees and sort
        cur_degs = np.sum(cur_graph*(1-np.eye(cur_graph.shape[0])), axis=0)
        temp = [(cur_degs[i], cur_vs[i]) for i in range(cur_graph.shape[0])]
        temp.sort()

        # Get everytthing but least connected v
        least_v = temp[0][1]
        idxs = np.where(cur_vs!=least_v)[0]

        # Remove least connectedv
        cur_graph = cur_graph[idxs, :][:, idxs]
        cur_vs = cur_vs[idxs]

        # Calcuate density
        data.append((cur_graph, cur_vs, density2(cur_graph)))
    data = [i for i in data if density(i[0])>th]
    data.sort(key=lambda x:x[2], reverse=True)
    if len(data)>0:
        return data[0]
    else:
        return None, None, None

def find_blocks2(g, vs=None, visualize=False, th=0.8):
    if vs is None:
        vs = np.arange(g.shape[0])
    cur_g = g
    cur_vs = vs
    outputs = []
    while cur_g.shape != (0,0):
        #print(d, cur_g.shape, cur_vs)
        subgraph, subgraph_vs, d = charikar2(cur_g, cur_vs, th=0.8)
        if d is None:
            for i in cur_vs:
                outputs.append(np.array([i]))
            break
        keep_vs = np.array([v for v in cur_vs if not v in subgraph_vs])
        keep_inds = np.array([i for i in range(len(cur_vs)) if cur_vs[i] in keep_vs])
        remove_inds = np.array([i for i in range(len(cur_vs)) if not cur_vs[i] in keep_vs])
        
        if visualize:
            plt.figure()
            plt.subplot(131)
            plt.imshow(cur_g, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            t = np.zeros(cur_g.shape)
            for i in remove_inds:
                for j in remove_inds:
                    t[i,j] = 1
            plt.subplot(132)
            plt.imshow(t, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            plt.subplot(133)
            plt.imshow(subgraph, vmin=0, vmax=1)
            plt.yticks([],[])
            plt.xticks([],[])
            plt.show()

        cur_g = cur_g[keep_inds, :][:, keep_inds]
        cur_vs = keep_vs
        outputs.append(subgraph_vs)
    return outputs

def estogram2sub(estogram,SS3,ulrs,opt):
    x = estogram ##just alias
    x = (x+np.transpose(x, [1,0,2]))/2
    b = binrize_estogram(x, exclude=ulrs, threshold = opt.subdef_conserve)
    nres = len(x)
    MINCONFCUT = 0.5 #lower confidence threshold until this value

    confcut = opt.subdef_confidence
    ncover_min = int(opt.min_coverage*nres)
    covered = [False for k in range(nres)]

    while True:
        g = b>confcut
        # mask covered as well
        #for i in covered: g[i] = False
        #print( "count: ", g.count(True) )
        
        if opt.algorithm == 1:
            outputs = find_blocks(g, visualize=False)
        else:
            outputs = find_blocks2(g, visualize=False, th=opt.density)

        # get SS-extended chunk list
        subdefs = []
        
        covered_tmp = copy.copy(covered)
        for isub,sub in enumerate(outputs):
            if len(sub) < 3: continue
            chunks = myutils.list2part(list(sub))
            length = [len(chunk) for chunk in chunks]
            ind_by_len = np.flip(np.argsort(length),0)
            
            subdef = []
            for ichunk in ind_by_len:
                chunk = chunks[ichunk]
                SSchunk = chunk2SS(chunk,SS3)
                if len(SSchunk) < 3: continue #at least one strand

                # exclude if any residue belongs to claimed one
                if [covered_tmp[i] for i in SSchunk].count(True) > 3: continue
                
                subdef.append(SSchunk)
                for i in SSchunk: covered_tmp[i] = True

            #append only if has more than one
            if len(subdef) > 1:
                subdefs.append(subdef)
                ## report
                if opt.debug:
                    print( "Sub %d: %d res"%(isub,sum([len(SSchunk) for SSchunk in subdef])) )
                for i,SSchunk in enumerate(subdef):
                    if opt.debug: print( "SSchunk %d: %d-%d"%(i, SSchunk[0], SSchunk[-1]) )
                    for res in SSchunk: covered[res] = True

        #if opt.debug: print( "Confcut, covered: %.2f %3d"%(confcut,covered.count(True)))
        
        if covered.count(True) >= ncover_min or confcut <= MINCONFCUT: break

        # make it loose
        confcut -= 0.1

    if confcut > MINCONFCUT:
        print( "Convereged at confcut %.2f, coverage/threshold: %4.2f/%4.2f"%(confcut, float(covered.count(True))/nres, float(ncover_min)/nres))
    else:
        print( "Terminated at confcut %.2f, coverage/threshold: %4.2f/%4.2f"%(confcut, float(covered.count(True))/nres, float(ncover_min)/nres))
        
    if opt.outfile != None:
        f = open(opt.outfile, "w")
        for i,subdef in enumerate(subdefs):
            f.write("Sub %d: %d chunks\n"%(i,len(subdef)))
            for j,chunk in enumerate(subdef):
                f.write("Chunk %d: "%j+" %d-%d\n"%(chunk[0],chunk[-1]))
        f.close()

    return subdefs

# Use this main function for short CAdev/ULR analysis...
def main(pose,opt,SS3=[]):
    data = np.load(opt.npz)
    estogram = data['estogram']
    Qres = data['lddt']

    add_default_options_if_missing(opt)

    if SS3 == []:
        dssp = PR.rosetta.core.scoring.dssp.Dssp(pose)
        dssp.insert_ss_into_pose( pose );
        SS3 = [pose.secstruct(ires) for ires in range(1,pose.size()+1)]
    
    # 1. Predict ULR
    if opt.debug: print("\n[estogram2FT] ========== ULR prediction ========")
    if opt.ulrs != []:
        ulrs = opt.ulrs
    else:
        ulrs, Qres_corr = estogram2ulr(estogram,opt)
        #opt.ulrmode = 'dynamic' #if want to try aggressive mode
        #ulr_aggr, _ = estogram2ulr(estogram,opt)

    # 1-a. assign maximum allowed deviation in coordinates
    maxdev = np.zeros(len(estogram))
    for i,val in enumerate(Qres_corr):
        maxdev[i] = Qres2dev(val) #fitting function
        
    if opt.debug:
        print( "ULR detected: ")
        for ulr in ulrs:
            ulrconf = np.mean(Qres[ulr[0]-1:ulr[-1]]) #confidence of current struct
            print( "%d-%d: confidence %.3f"%( ulr[0],ulr[-1],ulrconf) )

    # 2. Predict subs
    if opt.debug: print("\n[estogram2FT] ========== SubChunk assignment ========")
    subdef = estogram2sub(estogram,SS3,ulr,opt)
    
if __name__== "__main__":
    PR.init('-mute all')

    opt = arg_parser(sys.argv[1:])
    pose = PR.pose_from_file(opt.pdb)
    main(pose,opt)
