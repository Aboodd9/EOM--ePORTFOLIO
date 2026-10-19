#!/usr/bin/env python3
"""MovieLens comparison. Run after installing the supplied requirements."""
import matplotlib
matplotlib.use('Agg')

# Imports and reproducibility
from pathlib import Path
from urllib.request import urlretrieve
import zipfile, random, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option('display.max_columns', 30)
sns.set_theme(style='whitegrid')
print('NumPy:', np.__version__, '| pandas:', pd.__version__)

import os, json, hashlib, platform, sys, importlib.metadata
from IPython.display import display
OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
# Keep CPU scheduling comparable on small mini-batches.
os.environ.setdefault('OMP_NUM_THREADS','2')
_fig_no=0
_original_show=plt.show
def save_and_show(*args, **kwargs):
    global _fig_no
    _fig_no += 1
    plt.gcf().savefig(OUT/f'figure_{_fig_no:02d}.png',dpi=160,bbox_inches='tight')
    _original_show(*args, **kwargs)
    plt.close('all')
plt.show=save_and_show


# Download from the official GroupLens host (or reuse an existing local copy)
DATA_DIR = Path('data')
ROOT = DATA_DIR / 'ml-100k'
ZIP_PATH = DATA_DIR / 'ml-100k.zip'
URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

if not (ROOT / 'u.data').exists():
    DATA_DIR.mkdir(exist_ok=True)
    if not ZIP_PATH.exists():
        print('Downloading MovieLens 100K...')
        urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

assert (ROOT / 'u.data').exists(), 'Dataset not found. Place ml-100k inside ./data.'
print('Dataset ready at:', ROOT.resolve())

DATA_SHA256=hashlib.sha256((ROOT/'u.data').read_bytes()).hexdigest()
print('Rating-file SHA-256:',DATA_SHA256)


# Load ratings, users and items
ratings = pd.read_csv(ROOT/'u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])
users = pd.read_csv(ROOT/'u.user', sep='|', names=['user_id','age','gender','occupation','zip_code'])
item_cols = ['item_id','title','release_date','video_release_date','IMDb_URL'] + [
    'unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama',
    'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
items = pd.read_csv(ROOT/'u.item', sep='|', names=item_cols, encoding='latin-1')

print('Ratings shape:', ratings.shape, '| users:', users.shape[0], '| items:', items.shape[0])
display(ratings.head())
display(ratings['rating'].describe().to_frame().T)
print('Missing values in core rating fields:', ratings[['user_id','item_id','rating']].isna().sum().to_dict())
print('Duplicate user-item pairs:', ratings.duplicated(['user_id','item_id']).sum())


# Descriptive distributions only. Activity groups are defined after splitting.
user_counts=ratings.groupby('user_id').size()
fig,ax=plt.subplots(1,2,figsize=(12,4))
sns.countplot(data=ratings,x='rating',ax=ax[0],color='#4472C4')
ax[0].set_title('Observed rating distribution')
sns.histplot(user_counts,bins=30,ax=ax[1],color='#147D86')
ax[1].set_title('Ratings per user'); plt.tight_layout(); plt.show()


# Encode IDs and create temporal train/validation/test sets
u_values = np.sort(ratings.user_id.unique())
i_values = np.sort(ratings.item_id.unique())
u_map = {v:k for k,v in enumerate(u_values)}
i_map = {v:k for k,v in enumerate(i_values)}
ratings['u'] = ratings.user_id.map(u_map).astype(int)
ratings['i'] = ratings.item_id.map(i_map).astype(int)

ordered = ratings.sort_values(['user_id','timestamp','item_id'],kind='stable')
test_idx = ordered.groupby('user_id').tail(1).index
remaining = ordered.drop(test_idx)
val_idx = remaining.groupby('user_id').tail(1).index
train_idx = remaining.drop(val_idx).index
train, val, test = (ratings.loc[x].copy() for x in [train_idx, val_idx, test_idx])

n_users, n_items = len(u_values), len(i_values)
global_mean = train.rating.mean()
print({'train':len(train), 'validation':len(val), 'test':len(test),
       'users':n_users, 'items':n_items, 'train_mean':round(global_mean,3)})
assert set(train.index).isdisjoint(val.index) and set(train.index).isdisjoint(test.index)

assert set(val.index).isdisjoint(test.index)
assert len(train)+len(val)+len(test)==len(ratings)
assert (ratings.rating.between(1,5)).all()
training_counts=train.groupby('user_id').size()
# Quantiles across users, not repeated rating rows.
activity_bins=pd.qcut(training_counts.rank(method='first'),q=3,labels=['low','medium','high'])
test['activity_group']=test.user_id.map(activity_bins)
train[['user_id','item_id','timestamp']].to_csv(OUT/'train_split.csv',index=False)
val[['user_id','item_id','timestamp']].to_csv(OUT/'validation_split.csv',index=False)
test[['user_id','item_id','timestamp']].to_csv(OUT/'test_split.csv',index=False)


class BiasedMF:
    def __init__(self, n_users, n_items, factors=20, lr=.01, reg=.03, epochs=25, patience=3, seed=42):
        self.n_users, self.n_items = n_users, n_items
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.patience, self.seed = epochs, patience, seed

    def fit(self, frame, valid=None, verbose=True):
        rng = np.random.default_rng(self.seed)
        self.mu = frame.rating.mean()
        self.seen_u=np.zeros(self.n_users,dtype=bool); self.seen_i=np.zeros(self.n_items,dtype=bool)
        self.seen_u[frame.u.unique()]=True; self.seen_i[frame.i.unique()]=True
        self.P = rng.normal(0, .1, (self.n_users, self.factors))
        self.Q = rng.normal(0, .1, (self.n_items, self.factors))
        self.bu = np.zeros(self.n_users); self.bi = np.zeros(self.n_items)
        rows = frame[['u','i','rating']].to_numpy()
        best, wait, self.history = np.inf, 0, []
        for epoch in range(self.epochs):
            rng.shuffle(rows)
            for u, i, r in rows:
                u, i = int(u), int(i)
                pu = self.P[u].copy(); qi = self.Q[i].copy()
                err = r - (self.mu + self.bu[u] + self.bi[i] + pu @ qi)
                self.bu[u] += self.lr * (err - self.reg*self.bu[u])
                self.bi[i] += self.lr * (err - self.reg*self.bi[i])
                self.P[u] += self.lr * (err*qi - self.reg*pu)
                self.Q[i] += self.lr * (err*pu - self.reg*qi)
            target = valid if valid is not None else frame
            score = rmse(target.rating, self.predict(target.u, target.i))
            self.history.append(score)
            if score < best - 1e-4:
                best, wait = score, 0
                state = [x.copy() if hasattr(x,'copy') else x for x in (self.P,self.Q,self.bu,self.bi)]
            else:
                wait += 1
                if wait >= self.patience: break
        self.P,self.Q,self.bu,self.bi = state
        if verbose: print(f'MF epochs={len(self.history)}, best validation RMSE={best:.4f}')
        return self

    def predict(self, users, items):
        u = np.asarray(users, dtype=int); i = np.asarray(items, dtype=int)
        out = self.mu + self.bu[u] + self.bi[i] + np.sum(self.P[u]*self.Q[i], axis=1)
        known=self.seen_u[u]&self.seen_i[i]
        fallback=self.mu+self.bu[u]*self.seen_u[u]+self.bi[i]*self.seen_i[i]
        return np.clip(np.where(known,out,fallback),1,5)

def rmse(y, pred): return mean_squared_error(y, pred) ** .5

started=time.perf_counter()
mf = BiasedMF(n_users, n_items).fit(train, val)
mf_seconds=time.perf_counter()-started


# If needed in a fresh notebook environment, uncomment:
# %pip install -q torch
try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    raise ImportError('PyTorch is required. Run `%pip install torch` and restart the kernel.') from e

torch.manual_seed(SEED)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb=24):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb)
        self.item_emb = nn.Embedding(n_items, emb)
        self.net = nn.Sequential(nn.Linear(2*emb,64), nn.ReLU(), nn.Dropout(.15),
                                 nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, u, i):
        z = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        return 1 + 4*torch.sigmoid(self.net(z).squeeze(1))

def loader(frame, batch=1024, shuffle=False):
    ds = TensorDataset(torch.tensor(frame.u.values, dtype=torch.long),
                       torch.tensor(frame.i.values, dtype=torch.long),
                       torch.tensor(frame.rating.values, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

@torch.no_grad()
def ncf_predict(model, frame):
    model.eval(); out=[]
    for u,i,_ in loader(frame, 4096):
        out.append(model(u.to(device),i.to(device)).cpu().numpy())
    pred=np.concatenate(out)
    if hasattr(model,'seen_u'):
        known=model.seen_u[frame.u.to_numpy()] & model.seen_i[frame.i.to_numpy()]
        pred=np.where(known,pred,model.fallback_mean)
    return pred

def fit_ncf(train_df, valid_df, epochs=20, patience=3, verbose=True):
    torch.manual_seed(SEED)
    model = NCF(n_users,n_items).to(device)
    model.seen_u=np.zeros(n_users,dtype=bool); model.seen_i=np.zeros(n_items,dtype=bool)
    model.seen_u[train_df.u.unique()]=True; model.seen_i[train_df.i.unique()]=True
    model.fallback_mean=float(train_df.rating.mean())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=1)
    loss_fn = nn.MSELoss(); best=np.inf; wait=0; history=[]
    for ep in range(epochs):
        model.train()
        for u,i,r in loader(train_df,1024,True):
            u,i,r = u.to(device),i.to(device),r.to(device)
            opt.zero_grad(); loss=loss_fn(model(u,i),r); loss.backward(); opt.step()
        score=rmse(valid_df.rating,ncf_predict(model,valid_df)); history.append(score); sched.step(score)
        if score < best-1e-4:
            best,wait,state=score,0,copy.deepcopy(model.state_dict())
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(state)
    if verbose: print(f'NCF epochs={len(history)}, best validation RMSE={best:.4f}')
    return model,history

started=time.perf_counter()
ncf,ncf_history = fit_ncf(train,val)
ncf_seconds=time.perf_counter()-started


def sampled_ndcg_at_k(model_predict_pairs, train_seen, test_df, k=10, negatives=100, seed=42):
    rng=np.random.default_rng(seed); scores=[]; all_items=np.arange(n_items)
    seen_by_user=train_seen.groupby('u').i.apply(set).to_dict()
    for row in test_df.itertuples():
        if row.rating < 4: continue
        unseen=np.setdiff1d(all_items, np.fromiter(seen_by_user.get(row.u,set())|{row.i},dtype=int))
        neg=rng.choice(unseen,size=min(negatives,len(unseen)),replace=False)
        cand=np.r_[row.i,neg]; users=np.repeat(row.u,len(cand))
        pred=model_predict_pairs(users,cand)
        rank=int(np.where(np.argsort(-pred)==0)[0][0])+1
        scores.append(1/np.log2(rank+1) if rank<=k else 0.0)
    return float(np.mean(scores)),len(scores)

mf_test_pred=mf.predict(test.u,test.i)
ncf_test_pred=ncf_predict(ncf,test)
seen=pd.concat([train,val])
mf_ndcg,n_eval=sampled_ndcg_at_k(mf.predict,seen,test)

def ncf_pairs(users,items):
    frame=pd.DataFrame({'u':users,'i':items,'rating':np.zeros(len(users))})
    return ncf_predict(ncf,frame)
ncf_ndcg,_=sampled_ndcg_at_k(ncf_pairs,seen,test)

results=pd.DataFrame({
    'Model':['Biased matrix factorisation','Neural collaborative filtering'],
    'Test RMSE':[rmse(test.rating,mf_test_pred),rmse(test.rating,ncf_test_pred)],
    'Sampled NDCG@10':[mf_ndcg,ncf_ndcg],
    'Parameters':[mf.P.size+mf.Q.size+len(mf.bu)+len(mf.bi),sum(p.numel() for p in ncf.parameters())]
})
display(results.style.format({'Test RMSE':'{:.4f}','Sampled NDCG@10':'{:.4f}','Parameters':'{:,}'}))
print('Eligible held-out relevant ratings used for NDCG:',n_eval)

results['Training seconds']=[mf_seconds,ncf_seconds]
results.to_csv(OUT/'model_comparison.csv',index=False)


# Outer folds remain independent of early-stopping decisions.
from sklearn.model_selection import train_test_split
development=pd.concat([train,val]).reset_index(drop=True)
kf=KFold(n_splits=3,shuffle=True,random_state=SEED)
cv=[]
for fold,(tr_idx,outer_idx) in enumerate(kf.split(development),1):
    inner_train_idx,inner_val_idx=train_test_split(tr_idx,test_size=.15,random_state=SEED+fold)
    tr,iv,outer=development.iloc[inner_train_idx],development.iloc[inner_val_idx],development.iloc[outer_idx]
    assert set(inner_train_idx).isdisjoint(outer_idx)
    assert set(inner_val_idx).isdisjoint(outer_idx)
    fold_mf=BiasedMF(n_users,n_items,epochs=25,patience=3,seed=SEED+fold).fit(tr,iv,False)
    fold_ncf,_=fit_ncf(tr,iv,epochs=20,patience=3,verbose=False)
    cv += [{'fold':fold,'model':'MF','RMSE':rmse(outer.rating,fold_mf.predict(outer.u,outer.i))},
           {'fold':fold,'model':'NCF','RMSE':rmse(outer.rating,ncf_predict(fold_ncf,outer))}]
cv=pd.DataFrame(cv)
cv_summary=cv.groupby('model').RMSE.agg(['mean','std'])
display(cv_summary.round(4)); cv.to_csv(OUT/'development_folds.csv',index=False)
cv_summary.to_csv(OUT/'development_summary.csv')
sns.barplot(data=cv,x='model',y='RMSE',errorbar='sd')
plt.title('Development outer-fold RMSE (mean and SD)'); plt.show()


# Training curves and observed-versus-predicted diagnostic
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(mf.history,label='MF'); ax[0].plot(ncf_history,label='NCF')
ax[0].set(xlabel='Epoch',ylabel='Validation RMSE',title='Early-stopping curves'); ax[0].legend()
ax[1].scatter(test.rating,ncf_test_pred,alpha=.25,s=12)
ax[1].plot([1,5],[1,5],'k--'); ax[1].set(xlabel='Observed rating',ylabel='NCF prediction',title='NCF test predictions')
plt.tight_layout(); plt.show()


# Local additive explanation for one MF prediction
example=test.iloc[0]; u,i=int(example.u),int(example.i)
parts=pd.Series({'global mean':mf.mu,'user bias':mf.bu[u],'item bias':mf.bi[i],
                 'latent interaction':mf.P[u]@mf.Q[i]})
display(parts.to_frame('contribution').style.format('{:.3f}'))
print('Movie:',items.set_index('item_id').loc[example.item_id,'title'],
      '| observed:',example.rating,'| predicted:',round(float(parts.sum()),3))
parts.plot(kind='bar',color=['#4472C4','#70AD47','#FFC000','#ED7D31'],title='MF prediction decomposition');
plt.axhline(0,color='black',lw=.8); plt.show()

print('Final clipped score:',float(mf.predict([u],[i])[0]))
parts.rename('contribution').to_csv(OUT/'mf_explanation.csv')


# Repeated permutation reliance diagnostics for both models.
importance=[]
for model_name,predict,original in [
    ('MF',lambda f:mf.predict(f.u,f.i),mf_test_pred),
    ('NCF',lambda f:ncf_predict(ncf,f),ncf_test_pred)]:
    baseline=rmse(test.rating,original)
    for feature in ['u','i']:
        for repeat in range(10):
            rng=np.random.default_rng(SEED+repeat)
            perm=test.copy(); perm[feature]=rng.permutation(perm[feature].values)
            importance.append({'model':model_name,'input':feature,'repeat':repeat,
                               'RMSE increase':rmse(test.rating,predict(perm))-baseline})
importance=pd.DataFrame(importance)
importance.to_csv(OUT/'permutation_repeats.csv',index=False)
importance_summary=importance.groupby(['model','input'])['RMSE increase'].agg(['mean','std'])
display(importance_summary)
importance_summary.to_csv(OUT/'permutation_summary.csv')
sns.barplot(data=importance,x='input',y='RMSE increase',hue='model',errorbar='sd')
plt.title('Reliance on identity inputs under permutation'); plt.show()


# Merge demographic attributes solely after prediction; they were not model inputs
audit=test[['user_id','activity_group','rating']].copy()
audit['MF_error']=(audit.rating.values-mf_test_pred)**2
audit['NCF_error']=(audit.rating.values-ncf_test_pred)**2
audit=audit.merge(users[['user_id','age','gender']],on='user_id',how='left')
audit['age_group']=pd.cut(audit.age,[0,24,34,49,120],labels=['<25','25–34','35–49','50+'])

def subgroup_table(column):
    return audit.groupby(column,observed=True).agg(
        n=('rating','size'), MF_RMSE=('MF_error',lambda x:np.sqrt(x.mean())),
        NCF_RMSE=('NCF_error',lambda x:np.sqrt(x.mean()))).reset_index()

for col in ['activity_group','gender','age_group']:
    print('\n',col)
    display(subgroup_table(col).round(4))

activity=subgroup_table('activity_group').melt(id_vars=['activity_group','n'],var_name='model',value_name='RMSE')
sns.barplot(data=activity,x='activity_group',y='RMSE',hue='model');
plt.title('RMSE disparity by historical user activity'); plt.show()

for col in ['activity_group','gender','age_group']:
    subgroup_table(col).to_csv(OUT/f'subgroups_{col}.csv',index=False)


# Standard thresholded relevance confusion rates on observed held-out ratings only.
# These cannot measure exposure to unrated films or the entire catalogue.
diagnostic=pd.DataFrame({'actual':test.rating.to_numpy(),'MF':mf_test_pred,'NCF':ncf_test_pred})
error_rates=[]
for name in ['MF','NCF']:
    positive=diagnostic.actual>=4; predicted=diagnostic[name]>=4
    tp=int((positive&predicted).sum()); fp=int((~positive&predicted).sum())
    fn=int((positive&~predicted).sum()); tn=int((~positive&~predicted).sum())
    error_rates.append({'model':name,'TP':tp,'FP':fp,'FN':fn,'TN':tn,
        'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan,
        'false_negative_rate':fn/(fn+tp) if fn+tp else np.nan})
error_rates=pd.DataFrame(error_rates); display(error_rates)
error_rates.to_csv(OUT/'relevance_error_rates.csv',index=False)


# Compact, automatically generated evidence table for presentation slides
winner_rmse=results.loc[results['Test RMSE'].idxmin(),'Model']
winner_ndcg=results.loc[results['Sampled NDCG@10'].idxmax(),'Model']
summary=pd.DataFrame({
    'Decision evidence':['Lowest test RMSE','Highest sampled NDCG@10','Validation method','XAI method','Key bias slices'],
    'Finding':[winner_rmse,winner_ndcg,'Per-user temporal test + 3-fold development sensitivity',
               'MF decomposition + NCF permutation importance','Activity, gender and age']})
display(summary)


# Save a reproducible demonstration and trained model artefacts.
demo=test.iloc[0]; demo_user=int(demo.u)
seen_items=set(pd.concat([train,val]).query('u == @demo_user').i)
candidates=np.array([j for j in range(n_items) if j not in seen_items])
demo_frame=pd.DataFrame({'u':demo_user,'i':candidates,'rating':0.0})
demo_frame['MF']=mf.predict(demo_frame.u,demo_frame.i)
demo_frame['NCF']=ncf_predict(ncf,demo_frame)
demo_frame['item_id']=[int(i_values[j]) for j in candidates]
demo_frame=demo_frame.merge(items[['item_id','title']],on='item_id',how='left')
display(demo_frame.nlargest(10,'MF')[['title','MF','NCF']])
demo_frame.nlargest(10,'MF')[['item_id','title','MF','NCF']].to_csv(OUT/'demo_top10_MF.csv',index=False)
demo_frame.nlargest(10,'NCF')[['item_id','title','MF','NCF']].to_csv(OUT/'demo_top10_NCF.csv',index=False)
demo_info={'user_id':int(demo.user_id),'item_id':int(demo.item_id),
           'observed':float(demo.rating),'MF':float(mf.predict([int(demo.u)],[int(demo.i)])[0]),
           'NCF':float(ncf_predict(ncf,test.iloc[[0]])[0])}
(OUT/'demo_prediction.json').write_text(json.dumps(demo_info,indent=2))
np.savez_compressed(OUT/'mf_weights.npz',P=mf.P,Q=mf.Q,bu=mf.bu,bi=mf.bi,
                    mu=mf.mu,seen_u=mf.seen_u,seen_i=mf.seen_i)
torch.save({'state_dict':ncf.state_dict(),'seen_u':ncf.seen_u.tolist(),
            'seen_i':ncf.seen_i.tolist(),'fallback_mean':ncf.fallback_mean,
            'n_users':n_users,'n_items':n_items,'embedding_dim':24},OUT/'ncf_weights.pt')
maps={'user_ids':[int(v) for v in u_values],'item_ids':[int(v) for v in i_values]}
(OUT/'identifier_maps.json').write_text(json.dumps(maps))
versions={name:importlib.metadata.version(name) for name in
          ['numpy','pandas','matplotlib','seaborn','scikit-learn','torch']}
manifest={'python':sys.version,'platform':platform.platform(),'packages':versions,
          'seed':SEED,'data_sha256':DATA_SHA256,'split':'per-user last two events',
          'train':len(train),'validation':len(val),'test':len(test),
          'ranking_candidates':'one positive plus up to 100 unlabelled items',
          'ranking_eligible_users':n_eval,'execution_status':'completed'}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Saved all outputs to',OUT.resolve())

