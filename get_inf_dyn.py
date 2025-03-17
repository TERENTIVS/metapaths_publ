# Templates required for metapath count extraction and node INF computation

# Length-2 metapaths
metapath_templ_l2 = '''MATCH path = {pattern}''' + '''
WHERE n_source.{node_id_field} = $head AND 
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([
n_source.{node_id_field},
n_1.{node_id_field},
n_target.{node_id_field}
])
collect([n_source.{node_id_field}, n_1.{node_id_field}]) AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_target.{node_id_field}]) AS r2_pairs, type(r2) AS r2
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs'''

# Length-3 metapaths
metapath_templ_l3 = '''MATCH path = {pattern}''' + '''
WHERE n_source.{node_id_field} = $head AND 
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([
n_source.{node_id_field},
n_1.{node_id_field},
n_2.{node_id_field},
n_target.{node_id_field}
])
collect([n_source.{node_id_field}, n_1.{node_id_field}]) AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_2.{node_id_field}]) AS r2_pairs, type(r2) AS r2,
collect([n_2.{node_id_field}, n_target.{node_id_field}]) AS r3_pairs, type(r3) AS r3
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs,
r3, r3_pairs'''

# Length-4 metapaths
metapath_templ_l4 = '''MATCH path = {pattern}''' + '''
WHERE n_source.{node_id_field} = $head AND
n_target.{node_id_field} = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([
n_source.{node_id_field},
n_1.{node_id_field},
n_2.{node_id_field},
n_3.{node_id_field},
n_target.{node_id_field}
])
collect([n_source.{node_id_field}, n_1.{node_id_field}]) AS r1_pairs, type(r1) AS r1,
collect([n_1.{node_id_field}, n_2.{node_id_field}]) AS r2_pairs, type(r2) AS r2,
collect([n_2.{node_id_field}, n_3.{node_id_field}]) AS r3_pairs, type(r3) AS r3,
collect([n_3.{node_id_field}, n_target.{node_id_field}]) AS r4_pairs, type(r4) AS r4
RETURN
metapath_count,
r1, r1_pairs,
r2, r2_pairs,
r3, r3_pairs
r4, r4_pairs'''


query_templates_234 = {2: metapath_templ_l2,
                       3: metapath_templ_l3,
                       4: metapath_templ_l4}