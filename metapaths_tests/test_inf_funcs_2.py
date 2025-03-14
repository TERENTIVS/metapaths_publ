import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from py2neo import Graph
import re
import socket
from unittest.mock import patch

# ultimately, basic funcs should be in one submodule and core funcs in another

def find_highest_rel(metapath: str, rel_prefix):
    
    '''Requires metapath edges to have been assigned "r" + relation number" names
    as formatted by metapath_gen()'''
        
    rel_tag = rf"{rel_prefix}(\d+)"  # Matches the prefix followed by one or more digits
    matches = re.findall(rel_tag, metapath)
    if matches:
        return max(map(int, matches))  # Convert matches to integers, find the max, and add the prefix
    else:
        return None


def test_find_highest_rel():

    sample_metapath = '(n_source:Gene)-[r1:Gene_Gene]-(n_1:Gene)-[r2:Gene_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)'

    result = find_highest_rel(sample_metapath, 'r')

    assert result == 3


def create_fstr_from_template(template, **kwargs):
  '''
  Creates a formatted string from a template string and keyword arguments.

  Args:
    template: The template string with placeholder variables in curly brackets.
    **kwargs: The variables to be substituted into the template.

  Returns:
    The formatted f-string.
  '''
  template = template.replace('\n', ' ')
  return template.format(**kwargs)


def test_create_fstr_from_template():
   
   sample_str_to_match = "My name is Alice"

   result = create_fstr_from_template("My name is {pattern}", pattern='Alice')

   assert sample_str_to_match == result


metapath_templ_l2 = '''MATCH path = {pattern}''' + '''
WHERE n_source.name = $head AND n_target.name = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([n_source.name, n_1.name]) AS r1_pairs, type(r1) AS r1,
collect([n_1.name, n_target.name]) AS r2_pairs, type(r2) AS r2
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs'''

metapath_templ_l3 = '''MATCH path = {pattern}''' + '''
WHERE n_source.name = $head AND n_target.name = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([n_source.name, n_1.name]) AS r1_pairs, type(r1) AS r1,
collect([n_1.name, n_2.name]) AS r2_pairs, type(r2) AS r2,
collect([n_2.name, n_target.name]) AS r3_pairs, type(r3) AS r3
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs,
r3, r3_pairs'''

query_templates = {2: metapath_templ_l2, 3: metapath_templ_l3}


def get_inf_dict_save(graph: Graph, query_templates_by_length: dict,
                      head_type: str, tail_type: str,
                      target_pairs: pd.DataFrame, metapath_feats: list, 
                      nodes_freq: dict, reltype_counts,
                      save=False, save_str=None):
    
    '''
    Extracts pooled (min, max, mean) Inverse Node Frequencies per metapath feature for specified head-tail pairs.
    Also records the number of unique nodes at each relation in the metapath instance set.
    Requires apoc extension for Neo4j
    Uses pickle to save to specified destination..
    '''

    if 'PairID' not in target_pairs.columns:
    
        # assign PairID as concatenation of head and tail IDs
        target_pairs = target_pairs.assign(PairID = lambda row: row[head_type] + '_' + row[tail_type])
    
    target_pairs_inf = {feat: {} for feat in metapath_feats }

    for feat in metapath_feats:

        print(f'Processing {feat}...')

        path_length = find_highest_rel(feat, 'r')

        metapath_query_template = query_templates_by_length[path_length]

        metapath_query = create_fstr_from_template(metapath_query_template, pattern=feat)

        rels = ['r' + str(rel_idx+1) for rel_idx in range((path_length))]

        for _, row in tqdm(target_pairs.iterrows()):

            target_pair_name = row['PairID']

            target_pair_data = graph.run(metapath_query, head = row[head_type], tail = row[tail_type]).data()

            if len(target_pair_data) == 0: # no paths found - assign 0 for every field

                target_pairs_inf[feat][target_pair_name] = {}
                target_pairs_inf[feat][target_pair_name]['metapath_count'] = 0
                
                for rel in rels:

                    target_pairs_inf[feat][target_pair_name][f'{rel}_num_unique_nodes'] = 0
                    target_pairs_inf[feat][target_pair_name][f'{rel}_min_Inf'] = 0
                    target_pairs_inf[feat][target_pair_name][f'{rel}_max_Inf'] = 0
                    target_pairs_inf[feat][target_pair_name][f'{rel}_mean_Inf'] = 0
            else:

                target_pair_data = target_pair_data[0]
                
                target_pairs_inf[feat][target_pair_name] = {}

                target_pairs_inf[feat][target_pair_name]['metapath_count'] = target_pair_data['metapath_count']

                for rel in rels:
                
                    rel_name = target_pair_data[f'{rel}']

                    uniq_rel_pairs = set(tuple(pair) for pair in target_pair_data[f'{rel}_pairs'])

                    rel_heads = set([pair[0] for pair in uniq_rel_pairs])

                    rel_heads_node_f = [nodes_freq[rel_head][rel_name] for rel_head in rel_heads] # degs of head nodes for this rel

                    rel_heads_inf = [np.log10(
                                        (reltype_counts[rel_name].values / # N; use .values to avoid returning an array
                                        rel_head_node_f))
                                        for rel_head_node_f in 
                                        rel_heads_node_f]

                    rel_tails = set([pair[1] for pair in uniq_rel_pairs])

                    rel_tails_node_f = [nodes_freq[rel_tail][rel_name] for rel_tail in rel_tails] # degs of tail nodes for this rel

                    rel_tails_inf = [np.log10((reltype_counts[rel_name].values / 
                                        rel_tail_node_f)) 
                                        for rel_tail_node_f in 
                                        rel_tails_node_f]

                    target_pairs_inf[feat][target_pair_name][f'{rel}_num_unique_nodes'] = len(rel_heads.union(rel_tails))
                    target_pairs_inf[feat][target_pair_name][f'{rel}_min_Inf'] = np.min(rel_heads_inf + rel_tails_inf)
                    target_pairs_inf[feat][target_pair_name][f'{rel}_max_Inf'] = np.max(rel_heads_inf + rel_tails_inf)
                    target_pairs_inf[feat][target_pair_name][f'{rel}_mean_Inf'] = np.mean(rel_heads_inf + rel_tails_inf)

        if save:
            try:
                with open(save_str, 'wb') as file:
                    pickle.dump(target_pairs_inf, file)
                
                print(f'Saved INF data for feature {feat}')

            except TypeError:
                raise TypeError('Ensure appropriate save_str is passed')
    
    return target_pairs_inf             
    
def load_pickle(loc_str):
    with open(loc_str, 'rb') as file:
       loaded_pickle = pickle.load(file)
    return loaded_pickle

def test_get_inf_dict_save():

    drkg_testv = Graph("bolt://localhost:7687")

    testv_reltype_counts_reindexed = load_pickle('pickles/drkg_testv_reltype_counts_reindexed.pkl')

    testv_nodes_freq = load_pickle('pickles/drkg_testv_nodes_freq_undir.pkl')

    test_ids = load_pickle('pickles/test_all_drkg.pkl')

    test_pair_ids = test_ids.loc[test_ids['Gene']=='Gene::2693'].loc[test_ids['ADR']=='ADR::10021097']

    test_feats = ['(n_source:Gene)-[r1:Gene_Compound]-(n_1:Compound)-[r2:Compound_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)']

    test_pair_node_ids = {'Gene_Compound':['Gene::2693', 
                                        'Compound::DB00640'],
                    'Compound_Gene':['Compound::DB00640', 
                                    'Gene::135', 'Gene::551',
                                    'Gene::7124','Gene::2030',
                                    'Gene::3558', 'Gene::183'],
                    'Gene_ADR':['Gene::135', 'Gene::551',
                                'Gene::7124','Gene::2030',
                                'Gene::3558', 'Gene::183', 
                                'ADR::10021097']
                        }


    N_GC = testv_reltype_counts_reindexed['Gene_Compound'].values

    N_CG = testv_reltype_counts_reindexed['Compound_Gene'].values

    if N_GC == N_CG:
        print(True)

    N_GA = testv_reltype_counts_reindexed['Gene_ADR'].values

    # r1
    test_pair_node_inf_N_GC = []

    test_pair_node_ids['Gene_Compound']
    testv_nodes_freq['Compound::DB00640']['Gene_Compound'] == testv_nodes_freq['Compound::DB00640']['Compound_Gene']

    for node in test_pair_node_ids['Gene_Compound']:
    
        test_pair_node_inf_N_GC.append(np.log10(N_GC / testv_nodes_freq[node]['Gene_Compound']))

    min_GC_inf = np.min(test_pair_node_inf_N_GC)
    max_GC_inf = np.max(test_pair_node_inf_N_GC)
    mean_GC_inf = np.mean(test_pair_node_inf_N_GC)

    # r2
    test_pair_node_inf_N_CG = []

    for node in test_pair_node_ids['Compound_Gene']:
    
        test_pair_node_inf_N_CG.append(np.log10(N_CG / testv_nodes_freq[node]['Compound_Gene']))

    min_CG_inf = np.min(test_pair_node_inf_N_CG)
    max_CG_inf = np.max(test_pair_node_inf_N_CG)
    mean_CG_inf = np.mean(test_pair_node_inf_N_CG)

    # r3
    test_pair_node_inf_N_GA = []

    for node in test_pair_node_ids['Gene_ADR']:
    
        test_pair_node_inf_N_GA.append(np.log10(N_GA / testv_nodes_freq[node]['Gene_ADR']))

    min_GA_inf = np.min(test_pair_node_inf_N_GA)
    max_GA_inf = np.max(test_pair_node_inf_N_GA)
    mean_GA_inf = np.mean(test_pair_node_inf_N_GA)

    test_pairs_inf_manual = {'r1_min_Inf': min_GC_inf, 'r1_max_Inf': max_GC_inf, 'r1_mean_Inf': mean_GC_inf,
                    'r2_min_Inf': min_CG_inf, 'r2_max_Inf': max_CG_inf, 'r2_mean_Inf': mean_CG_inf,
                    'r3_min_Inf': min_GA_inf, 'r3_max_Inf': max_GA_inf, 'r3_mean_Inf': mean_GA_inf} 

    test_pairs_inf_comput = get_inf_dict_save(drkg_testv, query_templates, 'Gene', 'ADR', test_pair_ids, test_feats, testv_nodes_freq, testv_reltype_counts_reindexed)

    assert list(test_pairs_inf_comput.keys()) == test_feats

    pair_name_comput = list(test_pairs_inf_comput[test_feats[0]].keys())[0]

    assert pair_name_comput == 'Gene::2693_ADR::10021097'

    for k, v in test_pairs_inf_manual.items():

        assert test_pairs_inf_manual[k] == test_pairs_inf_comput[test_feats[0]][pair_name_comput][k]


def create_index(r_max):
    """
    Creates an index list with entries for 'metapath_count' and metrics for 
    'r1' to 'r{r_max}', including 'num_unique_nodes', 'min_Inf', 'max_Inf', and 'mean_Inf'.

    Args:
        r_max: The maximum value of 'r' to include in the index.

    Returns:
        A list of index entries.
    """

    index = ['metapath_count']
    for r in range(1, r_max + 1):
        index.extend([f'r{r}_num_unique_nodes', f'r{r}_min_Inf', f'r{r}_max_Inf', f'r{r}_mean_Inf'])
    return index


def test_create_index():

    result = create_index(3)

    assert result == ['metapath_count', 
                       'r1_num_unique_nodes', 'r1_min_Inf', 'r1_max_Inf', 'r1_mean_Inf',
                        'r2_num_unique_nodes','r2_min_Inf', 'r2_max_Inf', 'r2_mean_Inf', 
                        'r3_num_unique_nodes','r3_min_Inf', 'r3_max_Inf', 'r3_mean_Inf']


def extract_feat_dfs(inf_dict):

    feat_dfs = {}
    
    for feat, feat_data in inf_dict.items():

        feat_df = pd.DataFrame(feat_data).transpose()

        feat_dfs[feat] = feat_df

    return feat_dfs # dictionary of dataframes


def test_extract_feat_dfs():

    drkg_testv = Graph("bolt://localhost:7687")

    testv_reltype_counts_reindexed = load_pickle('pickles/drkg_testv_reltype_counts_reindexed.pkl')

    testv_nodes_freq = load_pickle('pickles/drkg_testv_nodes_freq_undir.pkl')

    test_ids = load_pickle('pickles/test_all_drkg.pkl')

    test_pair_ids = test_ids.loc[test_ids['Gene']=='Gene::2693'].loc[test_ids['ADR']=='ADR::10021097']

    test_feats = load_pickle('pickles/feats_rn.pkl')

    test_pairs_inf = get_inf_dict_save(drkg_testv, query_templates, 'Gene', 'ADR', test_pair_ids, test_feats, testv_nodes_freq, testv_reltype_counts_reindexed)

    feat_dfs = extract_feat_dfs(test_pairs_inf)

    for feat, feat_dict in test_pairs_inf.items():

        assert feat_dfs[feat].notna().all().all()

        for pair_dict in feat_dict.values():

            print(type(pair_dict))

            assert set(pair_dict.keys()).difference(set(feat_dfs[feat].columns)) == set()


def apply_params_to_feat(feat_df, rels, **param_combo):

    path_count = feat_df['metapath_count']

    if param_combo['path_deflator_exp'] != None:

        exp = param_combo['path_deflator_exp']

        if exp == 0:

            path_count = path_count.apply(lambda x: 1 if x > 0 else 0)
    
        else:
            
            path_count = path_count ** exp

    if param_combo['inf_inflator'] != None:

        assert 'inf_pooling' in param_combo, "Pooling option ('min', 'max' or 'mean') required"

        pool_option = param_combo['inf_pooling']

        pool_cols = [f'{rel}_{pool_option}_Inf' for rel in rels]

        if param_combo['inf_inflator'] == 'sum':

            inf_inflator = np.sum(feat_df[pool_cols], axis=1)

        elif param_combo['inf_inflator'] == 'product':

            inf_inflator = np.product(feat_df[pool_cols], axis=1)

        else:

            raise ValueError('Aggregation must be either sum or product')

    else:

        inf_inflator = 1

    final_feature = path_count * inf_inflator

    return final_feature


def test_apply_params_to_feat():

    drkg_testv = Graph("bolt://localhost:7687")

    testv_reltype_counts_reindexed = load_pickle('pickles/drkg_testv_reltype_counts_reindexed.pkl')

    testv_nodes_freq = load_pickle('pickles/drkg_testv_nodes_freq_undir.pkl')

    test_ids = load_pickle('pickles/test_all_drkg.pkl')

    test_pair_ids = test_ids.loc[test_ids['Gene']=='Gene::2693'].loc[test_ids['ADR']=='ADR::10021097']

    test_feats = load_pickle('pickles/feats_rn.pkl') # load non-empty paths of length 2 and 3

    test_pairs_inf = get_inf_dict_save(drkg_testv, query_templates, 'Gene', 'ADR', test_pair_ids, test_feats, testv_nodes_freq, testv_reltype_counts_reindexed)

    feat_dfs = extract_feat_dfs(test_pairs_inf)

    for feat, feat_df in feat_dfs.items():

        sample_params = {'path_deflator_exp': 0.5, 'inf_inflator': 'product', 'inf_pooling': 'min'}

        path_length = find_highest_rel(feat, 'r')

        if path_length == 2:

            sample_params_manual = feat_df['metapath_count'] ** 0.5 * (feat_df['r1_min_Inf'] * feat_df['r2_min_Inf'])

        else:

            sample_params_manual = feat_df['metapath_count'] ** 0.5 * (feat_df['r1_min_Inf'] * feat_df['r2_min_Inf'] * feat_df['r3_min_Inf'])

        feat_rels = ['r' + str(l+1) for l in range(int(path_length))]

        sample_params_comput = apply_params_to_feat(feat_df, feat_rels, **sample_params)
        
        assert len(sample_params_manual) == len(sample_params_comput)
        
        for i, _ in sample_params_manual.iteritems():

            assert sample_params_manual[i] == sample_params_comput[i]


def check_graph_connection(host, port):
    """
    Checks if a connection can be established to the specified host and port.

    Args:
    host: The hostname or IP address of the Neo4j server.
    port: The port number on which Neo4j is running.

    Returns:
    True if the connection is successful, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)  # Set a timeout of 1 second
            sock.connect((host, port))
        return True
    except (socket.timeout, ConnectionRefusedError) as e:
        print(f"Error connecting to {host}:{port}: {e}")
        return False


@patch('socket.socket')

def test_connection_check(mock_socket):
    mock_sock1 = mock_socket.return_value.__enter__.return_value
    mock_sock1.connect.return_value = None
    result1 = check_graph_connection("localhost", 7687)
    assert result1 is True

    mock_sock2 = mock_socket.return_value.__enter__.return_value
    mock_sock2.connect.side_effect = socket.timeout
    result2 = check_graph_connection("localhost", 7687)
    assert result2 is False

    mock_sock3 = mock_socket.return_value.__enter__.return_value
    mock_sock3.connect.side_effect = ConnectionRefusedError
    result3 = check_graph_connection("localhost", 7687)
    assert result3 is False

'''Leaving out testing of apply_params_to_feats due to its simplicity'''