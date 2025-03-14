import itertools
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from py2neo import Graph
import re
import pytest


def remove_newlines(template):

    '''
    Creates metapath Cypher queries from templates
    formatted with newline chars.
    '''

    template = template.replace('\n', ' ')
    return template


remove_newlines(
'''MATCH path = test_pattern
WHERE n_source.name = $head AND
n_target.name = $tail
AND apoc.coll.duplicates(NODES(path)) = []
WITH
COUNT(path) AS metapath_count,
collect([
n_source.name,
n_1.name,
n_target.name
]) AS metapath_nodes,
collect([n_source.name, n_1.name])
AS r1_pairs, type(r1) AS r1,
collect([n_1.name, n_target.name])
AS r2_pairs, type(r2) AS r2
RETURN
metapath_count,
metapath_nodes,
r1, r1_pairs,
r2, r2_pairs''')


def create_fstr_from_template(template, **kwargs):

    '''
    Creates metapath Cypher queries from templates
    formatted with newline chars.
    '''

    template = template.replace('\n', ' ')
    return template.format(**kwargs)


def test_create_fstr_from_template():

    sample_fstr = '''MATCH path = test_pattern
    WHERE n_source.name = $head AND
    n_target.name = $tail
    AND apoc.coll.duplicates(NODES(path)) = []
    WITH
    COUNT(path) AS metapath_count,
    collect([
    n_source.name,
    n_1.name,
    n_target.name
    ]) AS metapath_nodes,
    collect([n_source.name, n_1.name])
    AS r1_pairs, type(r1) AS r1,
    collect([n_1.name, n_target.name])
    AS r2_pairs, type(r2) AS r2
    RETURN
    metapath_count,
    metapath_nodes,
    r1, r1_pairs,
    r2, r2_pairs'''

    test_fstr = '''MATCH path = {pattern}
    WHERE n_source.{node_id_field} = $head AND
    n_target.{node_id_field} = $tail
    AND apoc.coll.duplicates(NODES(path)) = []
    WITH
    COUNT(path) AS metapath_count,
    collect([
    n_source.{node_id_field},
    n_1.{node_id_field},
    n_target.{node_id_field}
    ]) AS metapath_nodes,
    collect([n_source.{node_id_field}, n_1.{node_id_field}])
    AS r1_pairs, type(r1) AS r1,
    collect([n_1.{node_id_field}, n_target.{node_id_field}])
    AS r2_pairs, type(r2) AS r2
    RETURN
    metapath_count,
    metapath_nodes,
    r1, r1_pairs,
    r2, r2_pairs'''

    assert remove_newlines(sample_fstr) == create_fstr_from_template(test_fstr, pattern='test_pattern',
                                                    node_id_field='name')


def cypher_triple_to_list(triples: list, directed=True):
    '''
    Converts Cypher pattern strings to a list of lists representing triple types.

    Args:
    pattern_string: A Cypher pattern string, e.g., 
                    "(:Film)-[:Released_in]->(:Year)"

    Returns:
    A list of lists, where each inner list represents a triple type:
    [["(:Film)", "-[:Released_in]->", "(:Year)"]]
    '''

    formatted_triples = []

    for triple_str in triples:

        triple_parts = triple_str.split('-')

        if len(triple_parts) != 3:

            raise ValueError("Invalid Cypher triple string. It should have 3 parts separated by '-'.")
        
        triple_trimmed = []

        for part in triple_parts:

            part_trimmed = part
            
            if part[0]=='>':

                part_trimmed = part[1:]
                
            elif part[-1] == '<':

                part_trimmed = part[:-1]
                
            triple_trimmed.append(part_trimmed)

        node1, relationship, node2 = triple_trimmed

        if directed:
        
            relationship = f"-{relationship}->"

        else:
        
            relationship = f"-{relationship}-"

        formatted_triples.append([node1, relationship, node2])

    return formatted_triples


test_triples = [
    "(:ADR)-[:ADR_Gene]->(:Gene)",
    "(:SideEffect)-[:SideEffect_ADR]->(:ADR)",
    "(:Symptom)-[:Symptom_Disease]->(:Disease)"
]


# Directed edges
def test1_gen_triple_list():

    formatted_triples = cypher_triple_to_list(test_triples)

    result = formatted_triples

    assert result == [["(:ADR)","-[:ADR_Gene]->","(:Gene)"],
                      ["(:SideEffect)","-[:SideEffect_ADR]->","(:ADR)"],
                      ["(:Symptom)","-[:Symptom_Disease]->","(:Disease)"]
                      ]


# Undirected edges
def test2_gen_triple_list():
    
    formatted_triples = cypher_triple_to_list(test_triples, directed=False)

    result = formatted_triples

    assert result == [["(:ADR)","-[:ADR_Gene]-","(:Gene)"],
                      ["(:SideEffect)","-[:SideEffect_ADR]-","(:ADR)"],
                      ["(:Symptom)","-[:Symptom_Disease]-","(:Disease)"]
                      ]


test_triples_2 = [
    "(:Gene)-[:Gene_ADR]-(:ADR)",
    "(:ADR)-[:ADR_Gene]-(:Gene)",
    "(:Gene)-[:Gene_Gene]-(:Gene)",
    "(:Gene)-[:Gene_Pathway]-(:Pathway)",
    "(:Pathway)-[:Pathway_Gene]-(:Gene)",
    "(:Gene)-[:Gene_MolecularFunction]-(:MolecularFunction)",
    "(:MolecularFunction)-[:MolecularFunction_Gene]-(:Gene)"
]


def add_rel_variables(pattern):
    rel_count = 1
    output = ""
    i = 0
    while i < len(pattern):
        if pattern[i] == '[' and pattern[i+1] == ':':
            output += f"[r{rel_count}:"
            rel_count += 1
            i += 1  # Skip the ':'
        else:
            output += pattern[i]
        i += 1
    return output


def test_add_rel_variables():

    sample_paths = ['()-[:Directed_by]-()-[:Written_by]-()',
                    '()-[:Produced]-()-[:Distributed_by]-()-[:Distributed]-()']
    
    sample_paths_with_rels = ['()-[r1:Directed_by]-()-[r2:Written_by]-()',
                    '()-[r1:Produced]-()-[r2:Distributed_by]-()-[r3:Distributed]-()']
    
    result = add_rel_variables(sample_paths) == sample_paths_with_rels


def metapath_gen(source: str, target: str, triple_types: list, length: int):
    '''
    Generates all possible metapaths given source and target node types,
    available triple types to traverse and path length.

    Args:
    source & target: Neo4j node type patterns i.e. '(:Actor)'
    triple types: Neo4j triple pattern as formatted by
    cypher_triple_to_list().
    length: specifies the length of the generated metapath.
    '''

    def add_rel_variables(pattern):
        rel_count = 1
        output = ""
        i = 0
        while i < len(pattern):
            if pattern[i] == '[' and pattern[i+1] == ':':
                output += f"[r{rel_count}:"
                rel_count += 1
                i += 1  # Skip the ':'
            else:
                output += pattern[i]
            i += 1
        return output

    paths = []

    # generate permutations
    for p in itertools.product(triple_types, repeat=length):

        # check if starts and ends as desired
        if p[0][0] == source and p[-1][-1] == target:
            paths.append(p)

            # check if first node of next triple is always same as
            # last node of preceding triple
            for i in range(len(p)-1):
                if p[i+1][0] != p[i][-1]:
                    paths.remove(p)
                    break

    neo4j_paths = []
    for path in paths:

        flat = list(itertools.chain.from_iterable(path))
        # flatten list

        indices_to_pop = list(range(3, len(flat), 3))
        # get duplicates - every 4th node in list

        for i in sorted(indices_to_pop, reverse=True):
            flat.pop(i)
            # remove indices from reverse

        flat[0] = '(n_source' + flat[0][1:]
        flat[-1] = '(n_target' + flat[-1][1:]

        node_counter = 0
        for i, thing in enumerate(flat):
            if thing != flat[0] and thing != flat[-1] and thing[0] == '(':
                node_counter += 1
                flat[i] = '(n_' + str(node_counter) + thing[1:]

        # build continuous string from flattened list
        neo4j_path = ''

        for i in flat:
            neo4j_path += i
        neo4j_paths.append(neo4j_path)

    for i, path in enumerate(neo4j_paths):
        neo4j_paths[i] = add_rel_variables(path)
    return neo4j_paths


def test_metapath_gen():

    formatted_test_triples_2 = cypher_triple_to_list(test_triples_2, directed=False)
    
    result = set((metapath_gen('(:Gene)', '(:ADR)', formatted_test_triples_2, 3)))

    assert result == set(
                         ['(n_source:Gene)-[r1:Gene_Gene]-(n_1:Gene)-[r2:Gene_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)',
                          '(n_source:Gene)-[r1:Gene_Pathway]-(n_1:Pathway)-[r2:Pathway_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)',
                          '(n_source:Gene)-[r1:Gene_MolecularFunction]-(n_1:MolecularFunction)-[r2:MolecularFunction_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)',
                          '(n_source:Gene)-[r1:Gene_ADR]-(n_1:ADR)-[r2:ADR_Gene]-(n_2:Gene)-[r3:Gene_ADR]-(n_target:ADR)'
                          ]
    )


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
