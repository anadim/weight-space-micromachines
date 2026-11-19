import json, sys
# old = emulated, new = compiled
ROWS = ['pos','neg_nbr','neg_gen','neg_other','hard','ret','drift_doesF','pos_fire','hard_fire','neg_nbr_fire','ret_fire','mmlu_acc','mmlu_fire']
LBL = {'pos':'pos block','neg_nbr':'neighbour block','neg_gen':'general block','neg_other':'other-fence block','hard':'hard block',
       'ret':'retention block','drift_doesF':'drift: reply does topic','pos_fire':'pos fire','hard_fire':'hard fire',
       'neg_nbr_fire':'neighbour fire','ret_fire':'retention fire','mmlu_acc':'MMLU acc','mmlu_fire':'MMLU fire'}
for F in ['math','ml']:
    old = json.load(open(f'res/test_q17_{F}_summary.json'))
    new = json.load(open(f'res/test_q17c_{F}_summary.json'))
    print(f'\n===== fence {F}: OLD emulated vs NEW compiled (block/fire/acc %) =====')
    for c in ['probe_text_stop','monitor']:
        k = f'{F}/{c}'
        if k not in new: continue
        print(f'\n-- {c} --   {"metric":22s} {"OLD":>8} {"NEW":>8}')
        for r in ROWS:
            o = old.get(k,{}).get(r); n = new.get(k,{}).get(r)
            if o is None and n is None: continue
            fo = f'{100*o[0]:.1f}' if o else '-'; fn = f'{100*n[0]:.1f}' if n else '-'
            print(f'   {LBL.get(r,r):22s} {fo:>8} {fn:>8}')
        if 'silent' in new.get(k,{}): print('   silent (checked, identical, max|diff|):', new[k]['silent'])
