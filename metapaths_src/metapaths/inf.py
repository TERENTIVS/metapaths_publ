import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from py2neo import Graph
import socket
from .starterpack import save_pickle
from .starterpack import find_highest_rel, create_fstr_from_template


# Templates required for metapath count extraction and node INF computation

# Length-2 metapaths
metapath_templ_l2 = '''MATCH path = {pattern}
WHERE n_source.{node_id_field} = $head AND
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([n_source.{node_id_field}, n_1.{node_id_field}])
AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_target.{node_id_field}])
AS r2_pairs, type(r2) AS r2
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs'''

# Length-3 metapaths
metapath_templ_l3 = '''MATCH path = {pattern}
WHERE n_source.{node_id_field} = $head AND
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([n_source.{node_id_field}, n_1.{node_id_field}]) 
AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_2.{node_id_field}]) 
AS r2_pairs, type(r2) AS r2,
collect([n_2.{node_id_field}, n_target.{node_id_field}]) 
AS r3_pairs, type(r3) AS r3
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs,
r3, r3_pairs'''

# Length-4 metapaths
metapath_templ_l4 = '''MATCH path = {pattern}
WHERE n_source.{node_id_field} = $head AND
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([n_source.{node_id_field}, n_1.{node_id_field}]) 
AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_2.{node_id_field}]) 
AS r2_pairs, type(r2) AS r2,
collect([n_2.{node_id_field}, n_3.{node_id_field}]) 
AS r3_pairs, type(r3) AS r3,
collect([n_3.{node_id_field}, n_target.{node_id_field}]) 
AS r4_pairs, type(r4) AS r4
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs,
r3, r3_pairs
r4, r4_pairs'''


query_templates_234 = {2: metapath_templ_l2,
                       3: metapath_templ_l3,
                       4: metapath_templ_l4}

rel_data_to_get = [
                '{_rel}_min_Inf',
                '{_rel}_max_Inf',
                '{_rel}_mean_Inf'
]


def add_unseen_node_freq(graph, nodes, node_id_field,
                         nodes_freqs, rel_type):

    node_freq_query = f'''
                MATCH (n)-[rel]-()
                WHERE n.{node_id_field} = $node AND type(rel) = $rel
                RETURN COUNT(rel)
                '''
    for node in nodes:

        if node not in nodes_freqs:

            nodes_freqs[node] = {rel_type: graph.run(
                          node_freq_query,
                          node=node_id_field,
                          rel=rel_type).evaluate()
            }

        elif rel_type not in nodes_freqs[node]:

            nodes_freqs[node][rel_type] = graph.run(
                          node_freq_query,
                          node=node_id_field,
                          rel=rel_type).evaluate()


class INFToolbox:

    def __init__(self, graph: Graph, query_templates: dict):

        '''
        Class containing methods for the steps to produce
        Inverse Node Frequency transformations of the metapath count features
        of a given set of knowledge graph triples.
        Requires active Neo4j graph database connection via py2neo.'''

        self.graph = graph
        self._metapath_query_templates = query_templates
        self._param_combos = []

    @property
    def metapath_query_templates(self):
        return self._metapath_query_templates

    def check_graph_connection(self, host, port):
        '''
        Checks if a connection can be established to the specified host & port.
        Retuns True if the connection is successful, False otherwise.
        '''
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)  # Set a timeout of 1 second
                sock.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError) as e:
            print(f"Error connecting to {host}:{port}: {e}")
            return False

    def get_reltype_counts(self):

        '''
        Counts instances of relation types in the graph.
        Undirected edges are counted once.'''

        rel_type_instances_query = '''
        MATCH ()-[rel]->()
        RETURN DISTINCT(type(rel)), COUNT(rel)
        '''

        self.rel_type_counts = pd.DataFrame(
            self.graph.run(
                rel_type_instances_query
                )
                ).rename(
                    columns={0: 'Relation_type',
                                1: 'Count'}
                    )

        self.rel_type_counts_reindexed = self.rel_type_counts.set_index(
                                            'Relation_type').transpose()

        return self.rel_type_counts, self.rel_type_counts_reindexed

    def get_nodes_freq_dict(self, node_id_field: str,
                            reltype_counts: pd.DataFrame,
                            start_node_idx: int = None,
                            end_node_idx: int = None):

        '''Return dictionary of the graph nodes' relation-specific degrees.

        Args:
        node_id_field: the node property used as identifier.
        reltype_counts: output with index 0 from get_reltype_counts()
        start_ and end_node_idx: for use in testing on subset of nodes.
        '''

        nodes_query = f"MATCH (n) RETURN n.{node_id_field}"

        nodes_df = pd.DataFrame(self.graph.run(nodes_query).data())

        nodes_docf = {node: {} for node in nodes_df[f"n.{node_id_field}"].iloc[
                                                                start_node_idx:
                                                                end_node_idx]}

        node_freq_query = f'''
                        MATCH (n)-[rel]-()
                        WHERE n.{node_id_field} = $node AND type(rel) = $rel
                        RETURN COUNT(rel)
                        '''
        # not making a distinction between out-degree and in-degree as
        # any node type will typically be heads-only (all out-degree) or
        # tails-only (all in-degree)

        for node_id in tqdm(nodes_docf, total=len(nodes_docf)):

            for rel_type in reltype_counts['Relation_type']:

                nodes_docf[node_id][rel_type] = self.graph.run(
                    node_freq_query,
                    node=node_id,
                    rel=rel_type).evaluate()

        self.nodes_docf = nodes_docf
        
        return self.nodes_docf

    def get_inf_dict_save(self, target_pairs: pd.DataFrame,
                          head_type: str, tail_type: str,
                          metapath_feats: list,
                          node_id_field_: str,
                          reltype_counts_reindexed: dict,
                          on_demand_node_freqs: bool = True,
                          node_freqs_lookup: dict = None,
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
        head_type: name of head nodes' ID column
        (e.g. 'HEAD' or the node label).
        tail_type: name of tail nodes' ID column
        (e.g. 'TAIL' or the node label).
        metapath_feats: list of metapath features as formatted using
        metapath_featset_gen().
        node_id_field: Neo4j node property used as unique ID.
        reltype_counts_reindexed: output 1 from get_reltype_counts(). Required
        in INF calculation.
        on_demand_node_freqs: whether to compute per-relation degrees
        on demand; leave as True if node_freqs_lookup hasn't been computed in
        advance using get_nodes_freq_dict(). Otherwise set to False and provide
        node_freqs_lookup.

        If save=True, uses save_pickle() to save to specified
        destination.

        Returns nested dict with key hierarchy
        metapathfeature->nodepair->inf_data
        '''

        # assign PairID column as concatenation of head and tail IDs
        target_pairs = target_pairs.assign(PairID=lambda row: row[head_type] +
                                           '_' + row[tail_type])

        target_pairs_inf = {feat: {} for feat in metapath_feats}

        if on_demand_node_freqs is True:

            # initialise rel-specific deg lookup
            nodes_freqs = {}

        else:
            
            # use precomputed lookup
            nodes_freqs = node_freqs_lookup

        for feat in metapath_feats:

            print(f'Processing {feat}...')

            path_length = find_highest_rel(feat, 'r')

            metapath_query_template = self.metapath_query_templates[
                                                                path_length]

            metapath_query = create_fstr_from_template(
                metapath_query_template, pattern=feat,
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

                    for rel in rels:

                        rel_name = target_pair_data[f'{rel}']

                        uniq_rel_pairs = set(tuple(pair) for pair in
                                             target_pair_data[f'{rel}_pairs'])

                        rel_heads = set([pair[0] for pair in uniq_rel_pairs])

                        rel_tails = set([pair[1] for pair in uniq_rel_pairs])

                        rel_nodes = rel_heads | rel_tails

                        if on_demand_node_freqs is True:

                            # add rel-specific degs of any unseen nodes at this
                            # rel in the returned metapath instances
                            add_unseen_node_freq(self.graph,
                                                 rel_nodes,
                                                 node_id_field_,
                                                 nodes_freqs, rel_name)

                        # degs of nodes for this rel
                        rel_nodes_freqs = [nodes_freqs[rel_node][rel_name]
                                            for rel_node in rel_nodes]

                        # compute INF for nodes in rel
                        rel_nodes_inf = [np.log10(
                                            (reltype_counts_reindexed[
                                                rel_name].values /
                                                rel_node_freq)
                                                )
                                         for rel_node_freq
                                         in rel_nodes_freqs]
                        # use .values to avoid returning an array

                        # Pool INF values for this rel
                        target_pairs_inf[feat][target_pair_name][
                            f'{rel}_min_Inf'] = np.min(rel_nodes_inf)

                        target_pairs_inf[feat][target_pair_name][
                            f'{rel}_max_Inf'] = np.max(rel_nodes_inf)

                        target_pairs_inf[feat][target_pair_name][
                            f'{rel}_mean_Inf'] = np.mean(rel_nodes_inf)

        if save:
            try:
                save_pickle(target_pairs_inf, save_str)

            except TypeError:
                raise TypeError('Ensure appropriate save_str is passed')

        return target_pairs_inf

    def extract_feat_dfs(self, inf_dict):

        '''Extract a pandas DataFrame for each metapath feature, containing
        the data computed by get_inf_dict_save().
        Rows of the DataFrame correspond to the head-tail pairs and columns
        indicate the metapath feature's data.'''

        feat_dfs = {}

        for feat, feat_data in inf_dict.items():

            feat_df = pd.DataFrame(feat_data).transpose()

            feat_dfs[feat] = feat_df

        return feat_dfs  # dictionary of dataframes

    def apply_params_to_feats(self, feat_dfs, **param_combo):

        '''Apply INF parameterisation to metapath feature data as extracted by
        extract_feat_dfs().
        Returns a DataFrame with structure
        head-tail pairs * transformed features.'''

        def apply_params_to_feat(feat_df, rels, **param_combo):

            path_count = feat_df['metapath_count']

            if param_combo['path_deflator_exp'] is not None:

                exp = param_combo['path_deflator_exp']

                if exp == 0:

                    path_count = path_count.apply(lambda x: 1 if x > 0 else 0)

                else:

                    path_count = path_count ** exp

            if param_combo['inf_inflator'] is not None:

                assert 'inf_pooling' in param_combo, \
                    "Pooling option ('min', 'max' or 'mean') required"

                pool_option = param_combo['inf_pooling']

                pool_cols = [f'{rel}_{pool_option}_Inf' for rel in rels]

                if param_combo['inf_inflator'] == 'sum':

                    inf_inflator = np.sum(feat_df[pool_cols], axis=1)

                elif param_combo['inf_inflator'] == 'product':

                    inf_inflator = np.product(feat_df[pool_cols], axis=1)

                else:

                    raise ValueError(
                        'Aggregation must be either sum or product')

            else:

                inf_inflator = 1

            final_feature = path_count * inf_inflator

            return final_feature

        transformed_features = {}

        for feat, feat_df in tqdm(feat_dfs.items()):

            path_length = find_highest_rel(feat, 'r')

            feat_rels = ['r' + str(rel_idx+1) for rel_idx
                         in range(int(path_length))]

            transformed_feature = apply_params_to_feat(feat_df, feat_rels,
                                                       **param_combo)

            transformed_features[feat] = transformed_feature

        return pd.DataFrame(transformed_features)

    def add_param_combos(self, param_combos: list):

        '''Save parameter combinations to the toolbox. Use list with single
        element if adding a single parameter combination.'''

        self._param_combos.extend(param_combos)

    def run_pipeline(self, target_pairs: pd.DataFrame,
                     head_type: str, tail_type: str,
                     metapath_feats: list,
                     node_id_field_: str,
                     reltype_counts_reindexed: dict,
                     param_combos: list,
                     on_demand_node_freqs: bool = True,
                     node_freqs_lookup: dict = None,
                     save_inf_dict=False, save_str=None):
   
        '''
        Apply INF transformation according to the specified parameter
        combinations. Returns dict of (param_combo_idx, transformed_data)
        key-value pairs.

        Per-relation node degree policy is set on-demand as per default in
        get_inf_dict_save().

        Other args passed to internal get_inf_dict_save() call:
        target_pairs;
        head_type;
        tail_type;
        metapath_feats;
        node_id_field;
        reltype_counts_reindexed;
        save_inf_dict;
        save_str.

        Other args:
        param_combos: list of parameter combinations; each is a dict with
        format {'path_deflator_exp': float,
                'inf_inflator': {'sum','product'},
                'inf_pooling': {'min','max','mean'}}.

        Returns dict of (parameter_combination, INF-transformed
        metapath feature matrix as pd.DataFrame) key-value pairs.
        '''

        computed_inf_dict = self.get_inf_dict_save(target_pairs,
                                                   head_type, tail_type,
                                                   metapath_feats,
                                                   node_id_field_,
                                                   reltype_counts_reindexed,
                                                   on_demand_node_freqs,
                                                   node_freqs_lookup,
                                                   save_inf_dict, save_str)

        feat_dfs = self.extract_feat_dfs(computed_inf_dict)

        transformed_data = {}

        for combo_idx, combo in enumerate(param_combos):

            transformation = self.apply_params_to_feats(feat_dfs, **combo)

            transformed_data[combo_idx] = transformation

        return transformed_data
