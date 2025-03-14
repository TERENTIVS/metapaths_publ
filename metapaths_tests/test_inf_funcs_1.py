import itertools
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from py2neo import Graph
import pytest


def get_reltype_counts(graph: Graph):

    # always assume any undirected edge will be encoded with 2 unique labels
    rel_type_instances_query = '''
    MATCH ()-[rel]->()
    RETURN DISTINCT(type(rel)), COUNT(rel)
    '''

    rel_type_counts = pd.DataFrame(
        graph.run(
            rel_type_instances_query
            )
            ).rename(
                columns={0: 'Relation_type',
                            1: 'Count'}
                )
    
    rel_type_counts_reindexed = rel_type_counts.set_index('Relation_type').transpose()
    
    return rel_type_counts, rel_type_counts_reindexed

test_graph = Graph("bolt://localhost:7687") # console

# amend for own test_graph as needed
sample_reltype= 'Gene_ADR'

# amend for own test_graph as needed
sample_reltype_count_query = '''MATCH (:Gene)-[rel:Gene_ADR]->() RETURN COUNT(rel)'''


def test_get_reltype_counts():

    extracted_reltype_counts, _ = get_reltype_counts(test_graph)

    result = extracted_reltype_counts.loc[
             extracted_reltype_counts['Relation_type']==sample_reltype]['Count'].unique()
    
    sample_reltype_count = test_graph.run(sample_reltype_count_query
                                          ).evaluate()
    
    assert isinstance(sample_reltype_count, int)

    assert result == sample_reltype_count


def get_nodes_freq_dict(graph: Graph, node_id_field: str, reltype_counts: pd.DataFrame, 
                        start_node_idx: int = None, end_node_idx: int = None):
    
    '''Get dictionary of the graph nodes' relation-specific degrees.
    Requires output 0 from get_reltype_counts() i.e. not the reindexed counts.'''
    
    nodes_query = f"MATCH (n) RETURN n.{node_id_field}"

    nodes_df = pd.DataFrame(graph.run(nodes_query).data())

    nodes_docf = {node: {} for node in nodes_df[f"n.{node_id_field}"].
                  iloc[
                      start_node_idx:
                      end_node_idx]}

    # no need to specify direction as any node type will typically be only 
    # head or only tail 
    node_freq_query = f'''
    MATCH (n)-[rel]-()
    WHERE n.{node_id_field} = $node AND type(rel) = $rel
    RETURN COUNT(rel)
    '''

    for node_id in tqdm(nodes_docf, total=len(nodes_docf)):
        
        for rel_type in reltype_counts['Relation_type']:

            nodes_docf[node_id][rel_type] = graph.run(
                node_freq_query, 
                node=node_id, 
                rel=rel_type).evaluate()
            
    return nodes_docf


def test_get_nodes_freq_dict():

    sample_reltypes_df, _ = get_reltype_counts(test_graph)

    sample_rel_types = ['Gene_Compound', 'Compound_Gene']

    sample_node_id_field = 'name'

    sample_nodes_docf = get_nodes_freq_dict(test_graph, sample_node_id_field, sample_reltypes_df, end_node_idx=2)

    for node_id in sample_nodes_docf:

        for reltype in sample_rel_types:

            assert sample_nodes_docf[node_id][reltype] == test_graph.run(
                f"MATCH (n)-[rel:{reltype}]-() WHERE n.{sample_node_id_field} = '{node_id}' RETURN COUNT(rel)").evaluate()

