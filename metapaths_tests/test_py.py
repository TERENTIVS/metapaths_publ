from inf import INFToolbox, Graph, query_templates_234, load_pickle

toolbox = INFToolbox(Graph("bolt://localhost:7687"), query_templates_234)
toolbox.metapath_query_templates

testv_reltype_counts_reindexed = load_pickle(
    'pickles/drkg_testv_reltype_counts_reindexed.pkl')

testv_nodes_freq = load_pickle('pickles/drkg_testv_nodes_freq_undir.pkl')

test_ids = load_pickle('pickles/test_all_drkg.pkl')

test_feats = load_pickle('pickles/feats_rn.pkl')

trnsfrmd = toolbox.run_pipeline(test_ids, 'Gene', 'ADR', test_feats[:3], 
                            testv_nodes_freq, testv_reltype_counts_reindexed,
                            [{'path_deflator_exp': 0.5,
                              'inf_inflator': 'product',
                              'inf_pooling': 'min'}])

trnsfrmd[0]



