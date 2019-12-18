import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import argparse

# Estogram binarization
def binrize_estogram(estogram, threshold = 1):
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
    return output

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

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="A script to subdivide estograms via the charikar algorithm [2002]",
                                     epilog="v0.0.1")
    parser.add_argument("infile",
                        action="store",
                        help="input prediction .npz file")
    
    parser.add_argument("outfile",
                        action="store",
                        help="output .csv file")
    
    parser.add_argument("--conserve",
                        "-c", action="store",
                        type=float,
                        default=1.0,
                        help="threshold for errors [0.5, 1, 2, 4], default:1")
    
    parser.add_argument("--confidence",
                        "-conf", action="store",
                        type=float,
                        default=0.5,
                        help="threshold for confidence [0~1], default:0.5")
    
    parser.add_argument("--density",
                        "-d", action="store",
                        type=float,
                        default=0.9,
                        help="threshold for subgraph density [0~1], default:0.9")
    
    parser.add_argument("--algorithm",
                        "-a", action="store",
                        type=int,
                        default=0,
                        help="algorithm choice edge based/node based density, default (recommended): 0")
    
    args = parser.parse_args()
    
    data = np.load(args.infile)
    x = data["estogram"]
    x = (x+np.transpose(x, [1,0,2]))/2
    b = binrize_estogram(x, threshold = args.conserve)
    g = b>args.confidence
    if args.algorithm == 1:
        outputs = find_blocks(g, visualize=False)
    else:
        outputs = find_blocks2(g, visualize=False, th=args.density)
    f = open(args.outfile, "w")
    for i in outputs:
        f.write(",".join([str(j) for j in i])+"\n")
    f.close()
        
if __name__== "__main__":
    main()
