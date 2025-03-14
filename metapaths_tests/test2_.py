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




import pandas as pd
import numpy as np

def get_inf_dict_save(target_pairs: pd.DataFrame,
                        head_type: str, tail_type: str,
                        metapath_feats: list,
                        node_id_field_: str, reltype_counts: dict,
                        save=False, save_str=None):

    '''
    Extracts each metapath feature's Inverse Node Frequencies as pooled
    (min, max, mean) at each metapath relation type, for specified
    head-tail pairs representing the KG triple set of interest.
    Also records the number of unique nodes at each relation type in the
    metapath feature's instances.

    Requires apoc extension for Neo4j.

    Args:
    target_pairs: DataFrame with a row for every head-tail pair,
    containing at least the head and tail ID columns.
    head_type: name of head nodes' ID column.
    tail_type: name of tail nodes' ID column.
    metapath_feats: list of metapath features as formatted using
    metapath_featset_gen().
    nodes_freq: per-relation node degress as computed by
    get_nodes_freq_dict().
    reltype_counts: relation instance counts as computed by
    get_reltype_counts.
    If save=True, uses save_pickle() to save to specified
    destination.

    Returns nested dict with key tree structure
    metapathfeature->nodepair->inf_data
    '''

    # assign PairID column as concatenation of head and tail IDs
    target_pairs = target_pairs.assign(PairID=lambda row: row[head_type] +
                                        '_' + row[tail_type])

    target_pairs_inf = {feat: {} for feat in metapath_feats}

    # initialise rel-specific deg lookup here
    nodes_freqs = {}

    for feat in metapath_feats:

        print(f'Processing {feat}...')

        path_length = find_highest_rel(feat, 'r')

        metapath_query_template = self.metapath_query_templates[
                                                            path_length]

        metapath_query = create_fstr_from_template(metapath_query_template,
                                                    pattern=feat,
                                                    node_id_field=node_id_field_)

        rels = ['r' + str(rel_idx+1) for rel_idx in range((path_length))]

        for _, row in tqdm(target_pairs.iterrows()):

            target_pair_name = row['PairID']

            target_pairs_inf[feat][target_pair_name] = {}

            target_pair_data = self.graph.run(metapath_query,
                                                head=row[head_type],
                                                tail=row[tail_type]).data()

            if len(target_pair_data) == 0:

                # when no paths found assign 0 for every field
                target_pairs_inf[feat][target_pair_name][
                                                    'metapath_count'] = 0

                for rel in rels:
                    for item in rel_data_to_get:

                        target_pairs_inf[feat][target_pair_name][
                                create_fstr_from_template(item,
                                                            _rel=rel)] = 0

            else:

                target_pair_data = target_pair_data[0]

                target_pairs_inf[feat][target_pair_name][
                    'metapath_count'] = target_pair_data['metapath_count']

                metapath_nodes = set(target_pair_data['metapath_nodes'])

                for node in metapath_nodes:

                    if node not in nodes_freqs:

                        nodes_freqs[node] = get_node_freqs(
                            self.graph, node_id_field_, node,
                            reltype_counts)

                for rel in rels:

                    rel_name = target_pair_data[f'{rel}']

                    uniq_rel_pairs = set(tuple(pair) for pair in
                                            target_pair_data[f'{rel}_pairs'])

                    rel_heads = set([pair[0] for pair in uniq_rel_pairs])

                    # degs of head nodes for this rel
                    rel_heads_node_f = [nodes_freqs[rel_head][rel_name]
                                        for rel_head in rel_heads]

                    rel_heads_inf = [np.log10(
                                        (reltype_counts[rel_name].values /
                                            rel_head_node_f))
                                        for rel_head_node_f
                                        in rel_heads_node_f]
                    # use .values to avoid returning an array

                    rel_tails = set([pair[1] for pair in uniq_rel_pairs])

                    # degs of tail nodes for this rel
                    rel_tails_node_f = [nodes_freqs[rel_tail][rel_name]
                                        for rel_tail in rel_tails]

                    rel_tails_inf = [np.log10((reltype_counts[rel_name].
                                                values /
                                                rel_tail_node_f))
                                        for rel_tail_node_f
                                        in rel_tails_node_f]

                    target_pairs_inf[feat][target_pair_name][
                        f'{rel}_min_Inf'] = np.min(rel_heads_inf +
                                                    rel_tails_inf)

                    target_pairs_inf[feat][target_pair_name][
                        f'{rel}_max_Inf'] = np.max(rel_heads_inf +
                                                    rel_tails_inf)

                    target_pairs_inf[feat][target_pair_name][
                        f'{rel}_mean_Inf'] = np.mean(rel_heads_inf +
                                                        rel_tails_inf)

        if save:
            try:
                save_pickle(target_pairs_inf, save_str)
                print(f'Saved INF data for feature {feat}')

            except TypeError:
                raise TypeError('Ensure appropriate save_str is passed')

    return target_pairs_inf

get_inf_dict_save(test_ids, "Gene", "ADR", test_feats, "name", testv_reltype_counts_reindexed)