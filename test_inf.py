from metapaths.starterpack import load_pickle, save_pickle, cypher_triple_to_list, metapath_featset_gen
from metapaths.inf import INFToolbox, query_templates_234, Graph

tsa_dir = '/Users/terenceegbelo/Documents/PhD/Git/kg_sheffield/experiments/metapath_counts/TSA chapter 1/'

toolbox = INFToolbox(Graph("bolt://localhost:7687"), query_templates_234)
toolbox.add_param_combos([[{'path_deflator_exp': 0.5,
                            'inf_inflator': 'product',
                            'inf_pooling': 'min'}]])
toolbox.metapath_query_templates


testv_reltype_counts = load_pickle(
    tsa_dir+'pickles/drkg_testv_reltype_counts.pkl')


testv_reltype_counts_reindexed = load_pickle(
    tsa_dir+'pickles/drkg_testv_reltype_counts_reindexed.pkl')


test_ids = load_pickle(tsa_dir+'pickles/test_all_drkg.pkl')


test_feats = load_pickle(tsa_dir+'pickles/feats_rn.pkl')

#help(toolbox)

trnsfrmd = toolbox.run_pipeline(test_ids, 
                                'Gene', 'ADR', 
                                test_feats[:3],
                                'name',
                                testv_reltype_counts,
                                testv_reltype_counts_reindexed,
                                toolbox._param_combos[0])



