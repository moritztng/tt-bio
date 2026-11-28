import json,glob,os,statistics as st
O=os.path.expanduser('~/r4t/perf/k4')
def recs(n):
    return [json.loads(l) for l in open(f'{O}/{n}.txt') if l.startswith('{"step"')]
ok=True; fa=[];fb=[];clk=[]
for b in '012345678':
    A=recs(f'gA_0{b}')[0]; B=recs(f'gB_0{b}')[0]
    same=A['winner']==B['winner']==B['upstream_winner'] and A['master_digest']==B['master_digest']
    ok&=same; fa.append(A['fwd_s']); fb.append(B['fwd_s']); clk+=A['aiclk']+B['aiclk']
    print(f"b0{b} win {A['winner']}/{B['winner']}/{B['upstream_winner']} dig {A['master_digest']}/{B['master_digest']} fwd {A['fwd_s']:.3f}->{B['fwd_s']:.3f} {'OK' if same else 'DIFF'}")
X=recs('gX_00')[0]; disc=X['master_digest']!=recs('gB_00')[0]['master_digest']
print('(a)+(b)',ok,'| (c) gX differs',disc,X['master_digest'],'| fwd lower on',sum(b<a for a,b in zip(fa,fb)),'/9 | aiclk',min(clk),max(clk))
T={}
for arm in 'AB':
    for i in '123':
        try: r=recs(f't{arm}{i}')
        except FileNotFoundError: continue
        T.setdefault(arm,[]).append(r); clk+=[c for x in r for c in x['aiclk']]
for arm,runs in T.items():
    f=[x['fwd_s'] for r in runs for x in r[1:]]; s=[x['step_s'] for r in runs for x in r[1:]]; bw=[x['bwd_s'] for r in runs for x in r[1:]]
    print(arm,'n',len(f),'fwd med %.3f (%.3f-%.3f) bwd %.3f step %.3f'%(st.median(f),min(f),max(f),st.median(bw),st.median(s)),'digests',[x['master_digest'][:8] for x in runs[0]])
if 'A' in T and 'B' in T: print('timing digests identical', all(x['master_digest']==y['master_digest'] for ra in T['A'] for rb in T['B'] for x,y in zip(ra,rb)))
print('aiclk all', min(clk), max(clk))
