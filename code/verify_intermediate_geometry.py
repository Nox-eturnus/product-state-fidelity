"""Independent closed-form checks of the new diagnostics and moment arguments."""
import hashlib, json, math
from fractions import Fraction
from pathlib import Path
import numpy as np

def main():
    root=Path(__file__).resolve().parent
    d=json.loads((root.parent / "results" / "intermediate_geometry_results.json").read_text())
    rows=d['primary_rows']
    assert len(rows)==16 and len({r['row_id'] for r in rows})==16
    assert {r['method'] for r in rows}=={'native','shadow','naive','geometry','lp_common','lp_half','inv_common','inv_half'}
    checks=[]
    for r in rows:
        lam=float(r['target_id'].split('_')[1]);n=6;N=4096
        assert r['status']=='ok' and r['n']==n and r['N']==N and r['reps']==10000
        s=math.sqrt(lam/n);c=math.sqrt(1-lam/n);L=s+c
        means=np.array(r['rep_means_raw']);err=np.abs(np.clip(means,0,1)-.7)
        assert abs(float(np.quantile(err,.9))-r['p90'])<1e-13
        count=int(np.sum(err>.05));assert count==r['exceed_count']
        f=count/10000;z=1.96;den=1+z*z/10000
        mid=(f+z*z/20000)/den;half=z*math.sqrt(f*(1-f)/10000+z*z/(4*10000**2))/den
        assert np.allclose(r['exceed_wilson'],[max(0,mid-half),min(1,mid+half)],atol=1e-14)
        seeds=[int.from_bytes(hashlib.sha256(f"20260926|{r['target_id']}|n6|corr07|{r['method']}|4096|{i}".encode()).digest()[:16],'big') for i in range(10000)]
        assert len(set(seeds))==10000
        assert hashlib.sha256(','.join(map(str,seeds)).encode()).hexdigest()==r['rep_seeds_digest']
        method=r['method']
        if method=='native':
            mean=.7;second=.7*((3+L*L)/4)**n+.3*((L*L-1)/4)**n
            hi=((1+L)/2)**n;lo=(1-L)/2*((1+L)/2)**(n-1)
            assert abs(r['radius']-(hi-lo)*math.sqrt(math.log(6)/(2*N)))<1e-12
        elif method=='shadow':
            mean=.7;second=.7*1.5**n+.3*.5**n
        elif method=='naive':
            mean=.7*((1+c)/2)**n+.3*((1-c)/2)**n;second=mean
        else:
            half_signal=method.endswith('half');eta=(1-(.5 if half_signal else 1/math.sqrt(3)))/2
            if method=='geometry':assert r['D']==[] and r['k_proxy']==6
            if method.startswith('inv'):g=[float(Fraction(v)) for v in r['g_inv']]
            elif method=='geometry':
                g=[float(Fraction(v)) for v in next(a for a in rows if a['target_id']==r['target_id'] and a['method']=='lp_common')['g_round']]
            else:g=[float(Fraction(v)) for v in r['g_round']]
            probs=[math.comb(n,k)*(.7*eta**k*(1-eta)**(n-k)+.3*(1-eta)**k*eta**(n-k)) for k in range(n+1)]
            mean=sum(v*p for v,p in zip(g,probs));second=sum(v*v*p for v,p in zip(g,probs))
        assert abs(mean-r['expectation'])<1e-10 and abs(second-r['second_moment'])<1e-9
        se=math.sqrt((second-mean*mean)/(N*10000))
        assert abs(float(means.mean())-mean)<6*se
        checks.append({'row_id':r['row_id'],'closed_form_mean':mean,'closed_form_second_moment':second,'pass':True})
    # Direct 2x2 POVM summation, separate from the evaluator's scalar formulas.
    I=np.eye(2,dtype=complex);P=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    for theta in (.002,.2,.42):
        r=np.array([math.sin(theta),0,math.cos(theta)]);L=sum(abs(r));R=sum(r[i]*P[i] for i in range(3));M=np.zeros((2,2),complex)
        for i in (0,2):
            for o in (-1,1):M+=abs(r[i])/L*((1+L*o)/2)**2*(I+o*P[i])/2
        assert np.allclose(M,(1+L*L)/4*I+R/2)
    T=.25;z=(1-T*T)*np.exp(1j*T);t=1-z
    # At the Pauli endpoint: h_X=h_Y=1 and h_Z=1+t(o-1)/(2p_Z).
    uniform_second=2/3+abs(1-3*t)**2/3
    biased_second=.25+.75*abs(1-t/.75)**2
    assert uniform_second>1 and biased_second<1
    out={'passed':True,'rows':checks,'uniform_tau_minus_second_moment':float(uniform_second),'biased_tau_minus_second_moment':float(biased_second),'interpretation':'Native unclipped sample-mean MSE is (q^n-1)/N on the target; not a lower bound on clipped/high-probability estimators.'}
    (root.parent / "results" / "intermediate_geometry_checks.json").write_text(json.dumps(out,indent=2)+'\n')
    print('PASS: 16 rows; raw metrics, Wilson intervals, 128-bit seeds, independent closed-form moments, ranges and POVM identities.')
    print('Uniform and biased endpoint second moments:',uniform_second,biased_second)

if __name__=='__main__':main()
