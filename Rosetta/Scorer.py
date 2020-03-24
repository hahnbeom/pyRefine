import os,sys,copy
import numpy as np

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,SCRIPTDIR+'/../Critics')
print (sys.path)
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import CenQRunner
import pyrosetta as PR

class Scorer:
    def __init__(self,opt,cuts=[],normf=1.0):
        self.scoretype = opt.scoretype
        self.kT_mulfactor = 1.0 #??
        self.normf = normf

        self.wts = {}
        self.sfxn = None
        self.cenQscorer = None
        if self.scoretype.endswith(".wts"):
            self.init_from_wts_file(normf=normf)
        elif self.scoretype == 'Qcen': #Q-only
            self.wts['Qcen'] = 1.0
        elif self.scoretype == 'Qfa': #Q-only
            self.wts['Qfa'] = 1.0
        else:
            self.init_from_wts_file(normf=normf)
        print("SETUP Scorer, weights: ", self.wts)
        self.wts0 = copy.copy(self.wts)
        if self.sfxn != None: self.sfxn0 = self.sfxn.clone()
        
        if 'Qcen' in self.wts:
            if opt.dist != None:
                self.cenQscorer = CenQRunner.Scorer()
                self.dist_in = np.load(opt.dist)['dist'].astype(np.float32)
            else:
                self.cenQscorer = CenQRunner.Scorer(ver=0)
                self.dist_in = None
            # ignore neighboring residues to cut position
            chain_breaks = list()
            for res in cuts[:-1]: # final cut position = C-terminal
                chain_breaks.extend([res, res+1])
            self.chain_breaks = chain_breaks
            
        if 'dssp' in self.wts:
            self.Edssp = 1e6
            self.wdssp = 5.0
            self.ss8 = []
            if opt.ss_fn != None:
                self.ss8 = np.load(opt.ss_fn)['ss9']
                # corrections to termini
                self.ss8[0, :] = 1.0 / 8.0
                self.ss8[-1, :] = 1.0 / 8.0
                self.ss3 = np.zeros((len(self.ss8),3))
                #(0: B, 1: E, 2: U, 3: G, 4: H, 5: I, 6: S, 7: T, 8: C) 
                #BEGHIST_: H 2,3,4; E 0,1; L 5,6,7
                self.ss3[:,0] = np.sum([self.ss8[:,3],self.ss8[:,4],self.ss8[:,5]],axis=0) #H
                self.ss3[:,1] = np.sum([self.ss8[:,0],self.ss8[:,1],self.ss8[:,2]],axis=0) #E
                self.ss3[:,2] = np.sum([self.ss8[:,6],self.ss8[:,7],self.ss8[:,8]],axis=0) #C

    def close(self):
        if self.cenQscorer != None:
            self.cenQscorer.close()

    def init_from_wts_file(self,normf=1.0):
        self.wts['rosetta'] = 1.0*normf
        
        if self.scoretype == "default":
            self.sfxn = PR.create_score_function("score4_smooth")
            return

        self.sfxn = PR.create_score_function("empty")
        for l in open(self.scoretype):
            words = l[:-1].split()
            scoretype = words[0]
            wts = float(words[1])
            if scoretype == "Qcen":
                self.wts['Qcen'] = wts
            elif scoretype == "dssp":
                self.wts['dssp'] = wts*normf
            elif not scoretype.startswith("#"):
                st = PR.rosetta.core.scoring.score_type_from_name(scoretype)
                self.sfxn.set_weight(st, wts)
           
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

    def reset_scale(self,scoretype,scale):
        if scoretype in self.wts:
            self.wts[scoretype] = self.wts0[scoretype]*scale
        else:
            self.sfxn.set_weight(scoretype, scale*self.sfxn0[scoretype])
    
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
        is_single_input = False
        if not isinstance(poses,list):
            is_single_input = True
            poses = [poses]
            
        Es = np.zeros(len(poses))
        self.by_component = [{} for pose in poses] #cached every call
        
        if "Qcen" in self.wts and abs(self.wts['Qcen']) > 1.0e-6:
            CenQ_s = self.cenQscorer.score(poses, dist=self.dist_in, res_ignore=self.chain_breaks)
            Es += self.wts["Qcen"]*np.mean(CenQ_s,axis=1) # estimated residue-wise lddts

            for i,CenQ in enumerate(CenQ_s):
                self.by_component[i]['Qcen'] = self.wts["Qcen"]*np.mean(CenQ) #better keep as perres?
            
        for i,pose in enumerate(poses):
            if "dssp" in self.wts and self.ss8 != []:
                dsspscore = self.calc_dssp_agreement_score(pose,range(1,pose.size()))
                Es[i] += self.wts["dssp"]*dsspscore
                self.by_component[i]['dssp'] = self.wts["dssp"]*dsspscore
                
            if "rosetta" in self.wts:
                rosetta_score = self.sfxn.score(pose)
                emap = pose.energies().total_energies()
                cst = self.sfxn[PR.rosetta.core.scoring.atom_pair_constraint]*emap[PR.rosetta.core.scoring.atom_pair_constraint]
                chainbrk = self.sfxn[PR.rosetta.core.scoring.linear_chainbreak]*emap[PR.rosetta.core.scoring.linear_chainbreak]
                
                Es[i] += self.wts["rosetta"]*rosetta_score
                self.by_component[i]['rosetta'] = self.wts["rosetta"]*rosetta_score
                self.by_component[i]['chainbrk'] = self.wts["rosetta"]*chainbrk
                self.by_component[i]['cst'] = self.wts["rosetta"]*cst

        if is_single_input:
            return Es[0]
        else:
            return Es
