from metapaths.starterpack import load_pickle, save_pickle
from metapaths.inf import INFToolbox, query_templates_234, Graph

tsa_dir = '/Users/terenceegbelo/Documents/PhD/Git/kg_sheffield/experiments/metapath_counts/TSA chapter 1/'

toolbox = INFToolbox(Graph("bolt://localhost:7687"), query_templates_234)
toolbox
toolbox.add_param_combos([[{'path_deflator_exp': 0.5,
                            'inf_inflator': 'product',
                            'inf_pooling': 'min'}]])


testv_reltype_counts_reindexed = load_pickle(
    tsa_dir+'pickles/drkg_testv_reltype_counts_reindexed.pkl')


test_ids = load_pickle(tsa_dir+'pickles/test_all_drkg.pkl')


test_feats = load_pickle(tsa_dir+'pickles/feats_rn.pkl')

trnsfrmd = toolbox.run_pipeline(test_ids, 
                                'Gene', 'ADR', 
                                test_feats[:3],
                                'name',
                                testv_reltype_counts_reindexed,
                                toolbox._param_combos[0],
                                )

trnsfrmd[0].iloc[:5, 0]
#save_pickle(trnsfrmd, 'pickles/dyn_inf_17_03_25.pkl')


prev_trnsfrmd = load_pickle(tsa_dir + 
                            'pickles/inf_nodefreqlookup_testset.pkl')[0]

prev_trnsfrmd.iloc[:5, 0]
