import os,sys
import numpy as np

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,SCRIPTDIR+'/../Critics')
print (sys.path)
#sys.path.insert(0,'/home/hpark/NextGenSampler/pyrosetta/Critics')
import tensorflow as tf
import CenQRunner

class Scorer:
    def __init__(self,opt,cuts=[]):
        self.scoretype = opt.scoretype
        
        if self.scoretype == 'Q':
            if opt.dist != None:
                self.cenQscorer = CenQRunner.Scorer()
                self.dist_in = np.load(opt.dist)['dist'].astype(np.float32)
            else:
                self.cenQscorer = CenQRunner.Scorer(ver=0)
                self.dist_in = None
            #
            # ignore neighboring residues to cut position
            chain_breaks = list()
            for res in cuts[:-1]: # final cut position = C-terminal
                chain_breaks.extend([res, res+1])
            self.chain_breaks = chain_breaks
            
        else:
            sfxn = PR.create_score_function("score3")
            sfxn.set_weight(PR.rosetta.core.scoring.cen_hb, 5.0)
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

    def score(self,poses):
        if self.scoretype == "Q":
            CenQ_s = self.cenQscorer.score(poses, dist=self.dist_in, res_ignore=self.chain_breaks)
            #return CenQ_s
            # estimated residue-wise lddts
            CenQ_G = np.mean(CenQ_s,axis=1)
            return -CenQ_G #make it as "Energy"

        else:
            for pose in poses:
                self.Edssp = 0.0
                self.E = self.sfxn.score(pose)
                if self.ss8 != []:
                    #for res in range(pose.size()):
                    #TODO: score only ulr? 
                    self.Edssp += self.calc_dssp_agreement_score(pose,range(1,pose.size()))
                E = self.E + self.Edssp
                Es.append(E)
        return Es
