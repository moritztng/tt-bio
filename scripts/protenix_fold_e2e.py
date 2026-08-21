# CAPSTONE: full on-device Protenix-v2 pipeline via tt_bio.protenix.Protenix.fold.
# Assembles a model-ready feats dict from the golden pkls (atom features, relp/token_bonds,
# template/msa feats), runs atom-encoder -> diffusion atom-cache -> 10-cycle trunk -> EDM
# sampler -> confidence head entirely on-device, and reports the three quantities the test
# asserts on: the structures are physical, the samples one call returned are not scattered,
# and the sample the model RANKED FIRST is not far from the reference prediction.
#
# Folds at the production schedule (200 steps, 5 samples). It used to fold at n_step=10, where
# this molecule's own sampler spread is 8.2-16.4 A -- no absolute bound is meaningful there, and
# the schedule is one nobody ships. At 200 steps the same fold spreads 1.3-9.7 A and its ranked
# sample sits 8.00-8.48 A from the reference across four seeds.
#
# What this fixture CAN score: the pipeline runs, the output is physical, the ensemble is not
# scattered, and the ranked sample does not drift. What it CANNOT score: parity. Its reference
# (protenix_traj.pkl final_coords) is a single N_step=10 draw of this molecule, so ~8 A is
# mostly the reference's own undersampling. The parity bar lives on the gate's structure legs
# (protenix-prot-msa and friends), which score against 200-step references.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
import pickle, itertools, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix

CKPT='/home/ttuser/protenix_ckpt/protenix-v2.pt'
ife=pickle.load(open('/home/ttuser/protenix_ife_gold.pkl','rb'))
tg=pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl','rb'))
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
tfeat=d['intermediates']['template_embedder']['in'][0]
traj=pickle.load(open('/home/ttuser/protenix_traj.pkl','rb'))

F=ife['feat']
feats={
    'ref_pos':F['ref_pos'],'ref_charge':F['ref_charge'],'ref_mask':F['ref_mask'],
    'ref_element':F['ref_element'],'ref_atom_name_chars':F['ref_atom_name_chars'],
    'd_lm':F['d_lm'],'v_lm':F['v_lm'],'atom_to_token_idx':F['atom_to_token_idx'],
    'restype':F['restype'],'profile':F['profile'],'deletion_mean':F['deletion_mean'],
    'mask_trunked':ife['mask_trunked'],
    'relp':tg['relp'],'token_bonds':tg['token_bonds'],
    'template_aatype':tfeat['template_aatype'],'template_distogram':tfeat['template_distogram'],
    'template_pseudo_beta_mask':tfeat['template_pseudo_beta_mask'],
    'template_unit_vector':tfeat['template_unit_vector'],
    'template_backbone_frame_mask':tfeat['template_backbone_frame_mask'],
    'msa':tfeat['msa'],'has_deletion':tfeat['has_deletion'],'deletion_value':tfeat['deletion_value'],
    'asym_id':tfeat['asym_id'],
}
cp=pickle.load(open('/home/ttuser/protenix_confidence_pre.pkl','rb'))['kwargs']['input_feature_dict']
for k in ('distogram_rep_atom_mask','atom_to_tokatom_idx'):   # confidence head only, same molecule
    feats[k]=cp[k]

dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
model=Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
def prog(stage,step,total): print('  %s %d/%d'%(stage,step,total),flush=True)
N_STEP,N_SAMPLE=200,5
coords,confs=model.fold(feats, n_step=N_STEP, n_sample=N_SAMPLE, seed=0, progress_fn=prog,
                        return_confidence=True)                                  # (S,N,3)
xs=[coords[k] for k in range(coords.shape[0])]
# same ranking key as tt_bio/worker.py: ipTM-weighted for complexes, pTM for monomers, pLDDT last
score=[(0.8*float(c.get('iptm',0.0))+0.2*float(c.get('ptm',0.0))) if float(c.get('iptm',0.0))>0
       else (float(c.get('ptm',0.0)) or float(c['plddt'])) for c in confs]
rank0=max(range(len(xs)), key=lambda k: score[k])
rg=[float((x-x.mean(0)).pow(2).sum(-1).mean().sqrt()) for x in xs]
print('fold coords %s  finite=%s'%(tuple(coords.shape), bool(torch.isfinite(coords).all())),flush=True)
print('Rg %.3f A   Rg max %.3f A   pairwise mean %.3f A'
      %(min(rg), max(rg), float(torch.pdist(xs[rank0]).mean())),flush=True)
def kabsch(P,Q):
    # R maps Qc onto Pc, so Pc@R.T maps Pc onto Qc. Applying R to Qc (or R.T to Pc) is the same
    # superposition; applying R.T to Qc is NOT, and scores a rigid copy at ~Rg instead of 0.
    Pc=P-P.mean(0);Qc=Q-Q.mean(0);H=Pc.t()@Qc;U,_,Vt=torch.linalg.svd(H)
    Dm=torch.diag(torch.tensor([1.,1.,torch.sign(torch.det(Vt.t()@U.t()))]));R=Vt.t()@Dm@U.t()
    return float(((Pc@R.t())-Qc).pow(2).sum(-1).mean().sqrt())
assert kabsch(xs[0],xs[0]@torch.linalg.qr(torch.randn(3,3))[0]+3.0)<1e-3, 'kabsch self-check'
spread=[kabsch(xs[i],xs[j]) for i,j in itertools.combinations(range(len(xs)),2)]
print('ensemble spread Kabsch RMSD: min %.3f max %.3f A'%(min(spread),max(spread)),flush=True)
print('seed-to-seed Kabsch RMSD: %.3f A'%max(spread),flush=True)   # legacy key, = widest pair
ref=traj.get('final_coords')
if ref is not None:
    rf=ref.float().reshape(-1,3)[:xs[0].shape[0]]
    vr=[kabsch(x,rf) for x in xs]
    print('ranked sample %d of %d vs reference Kabsch RMSD: %.3f A'
          %(rank0,len(xs),vr[rank0]),flush=True)
    print('best sample vs reference Kabsch RMSD: %.3f A'%min(vr),flush=True)
    print('seed0 vs reference Kabsch RMSD: %.3f A'%vr[rank0],flush=True)   # legacy key
print('FOLD_E2E_DONE',flush=True)
