"""Print the top N rows of an analyze.py ops.json as a pipe table: shapes, memory, fidelity, chunks, roofline.

usage: table.py OPS.json SUMMARY.json N
"""
import json,re,sys
d=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2]))
def T(x):
    if isinstance(x,dict) and 'T' in x: yield x['T']
    elif isinstance(x,list):
        for y in x: yield from T(y)
    elif isinstance(x,dict):
        for y in x.values(): yield from T(y)
dn={'BFLOAT16':'bf16','FLOAT32':'fp32','BFLOAT8_B':'bfp8','BFLOAT4_B':'bfp4','UINT32':'u32'}
for r in d['ops'][:int(sys.argv[3])]:
    ins=list(T(r['args']))+list(T(r['kwargs']))
    sh=' x '.join('%s%s'%(t['shape'],dn.get(t['dtype'],t['dtype'])) for t in ins[:2])
    mem='L1' if any('L1' in t['mem'] for t in ins) else 'DRAM'
    pc=r['program_config'] or ''
    m=re.findall(r'(q_chunk_size=\d+|k_chunk_size=\d+|in0_block_w=\d+|per_core_M=\d+|per_core_N=\d+)',pc)
    print(f"{r['rank']}|{r['share']*100:.2f}|{r['op'].replace('ttnn.','').replace('transformer.scaled_dot_product_attention','sdpa')}|{r['cls'][:22]}|{r['call_site'][0]}|{sh}|{mem}|{r['math_fidelity'] or 'dflt'}/{(r['fp32_dest_acc'] or '-')[:1]}|{','.join(x.split('=')[0][:3]+'='+x.split('=')[1] for x in m)}|{r['calls_full_fold']:.0f}|{r['device_us_per_call']:.0f}|{r['achieved_tflops']:.1f}|{r['achieved_gbs']:.0f}|{r['roof']['fraction_of_roof']:.2f}|{r['bound']}")
