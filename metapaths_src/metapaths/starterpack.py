import itertools
import pickle
import re
import socket


def save_pickle(var_, loc_str):
    with open(loc_str, 'wb') as file:
        pickle.dump(var_, file)


def load_pickle(loc_str):
    with open(loc_str, 'rb') as file:
        loaded_pickle = pickle.load(file)
    return loaded_pickle


def check_graph_connection(host, port):

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


def cypher_triple_to_list(triples: list, directed=True):
    '''
    Converts a Cypher triple string to a list of strings
    representing the triple's head, relation, tail sequence for directed
    relationships, or the unordered sequence of entity, relation, entity
    for undirected relationships.

    This formatting is required for exhaustive metapath Cypher pattern
    generation from available triples using metapath_gen() and
    metapath_featset_gen().

    Args:
    triples: list of Cypher pattern strings, e.g.,
                    [
                    "(:Film)-[:Released_in]->(:Year)",
                    "(:Film)-[:Features]->(:Song)"
                    ]

    directed: specify if the formatting should add the ">"
    character to indicate a directed relationship.

    Returns:
    A list of lists, where each inner list represents a
    formatted triple type e.g.:
    [
    ["(:Film)", "-[:Released_in]->", "(:Year)"],
    ["(:Film)", "-[:Features]->", "(:Song)"]
    ]
    '''

    formatted_triples = []

    for triple_str in triples:

        triple_parts = triple_str.split('-')

        if len(triple_parts) != 3:

            raise ValueError(
                'Invalid Cypher triple string. It should have 3 \
                    parts separated by "-".')

        triple_trimmed = []

        for part in triple_parts:

            part_trimmed = part

            if part[0] == '>':

                part_trimmed = part[1:]

            elif part[-1] == '<':

                part_trimmed = part[:-1]

            triple_trimmed.append(part_trimmed)

        node1, relationship, node2 = triple_trimmed

        if directed:

            rel_w = [node1, f'-{relationship}->', node2]  # western
            rel_e = [node2, f'<-{relationship}-', node1]  # eastern

        else:

            rel_w = [node1, f'-{relationship}-', node2]  # western
            rel_e = [node2, f'-{relationship}-', node1]  # eastern

        formatted_triples.append(rel_w, rel_e)

    return formatted_triples


def add_rel_variables(pattern):

    '''
    Add r<rel_number> variable name to all edges/relations in a metapath
    Cypher pattern if missing e.g. if metapath was not generated using
    metapath_gen(). Relation variables are required for INF data processing.
    '''

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


def metapath_gen(source: str, target: str, triple_types: list, length: int):
    '''
    Generates all possible metapaths given source and target node types,
    available triple types to traverse and path length.

    Args:
    source & target: Neo4j node type patterns i.e. '(:Actor)'
    triple types: Neo4j triple pattern as formatted by
    cypher_triple_to_list().
    length: specifies the length of the generated metapaths.
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


def metapath_featset_gen(source: str, target: str,
                         triple_types: list, lengths: list):
    '''
    Generate list of metapaths in Neo4j query format for specified lenghts,
    using the lower-level function metapath_gen().

    Args:
    source and target: as required by metapath_gen().
    lengths: list of integers specifying lenghts of metapaths.
    triple_types: Neo4j triple pattern as formatted by
    cypher_triple_to_list().
    lengths: specifies the lengths of the generated metapaths.
    '''

    feats = []

    for length in lengths:

        feats.append(metapath_gen(source, target, triple_types, length))

    feats_flat = list(itertools.chain.from_iterable(feats))

    return feats_flat


def find_highest_rel(metapath: str, rel_prefix):

    '''Requires metapath edges to have been assigned "r" + relation number"
    names as formatted by metapath_gen() in the module "starterfuncs".'''

    rel_tag = rf"{rel_prefix}(\d+)"  # Matches the prefix followed by 1+ digits
    matches = re.findall(rel_tag, metapath)
    if matches:
        return max(map(int, matches))  # Convert matches to integers
    else:
        return None


def create_fstr_from_template(template, **kwargs):

    '''
    Creates metapath Cypher queries from templates formatted with newline chars
    e.g. re-formats metapaths saved with newlines for neatness.
    '''

    template = template.replace('\n', ' ')
    return template.format(**kwargs)
