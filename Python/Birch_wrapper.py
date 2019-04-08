from itertools import chain
from textwrap import dedent
from freediscovery.externals.jwzthreading import Container
from sklearn.exceptions import NotFittedError


class BirchSubcluster(Container):

    @property
    def cluster_item_count(self):
        """Count of all documents in the children subclusters"""
        partial_sum = sum([child.cluster_item_count for child in self.children])
        return len(self.get('document_id', [])) + partial_sum

    @property
    def cluster_id_accumulated(self):
        """Returns  list of document / sample ids contained
        in this subcluster or any of its children."""
        partial_sum = chain.from_iterable(el.cluster_id_accumulated
                                          for el in self.children)
        return list(self.get('document_id', [])) + list(partial_sum)

    def limit_depth(self, max_depth=None):
        """ Truncate the tree to the provided maximum depth

        Parameters
        ----------
        max_depth : int
          hierarchy depth to which truncate the tree
        """
        if self.current_depth >= max_depth:
            self.children = []

        for el in self.children:
            el.limit_depth(max_depth)

    def display_tree(self, max_depth=None):
        """Print the content of hierarchical tree below this subcluster
        """
        _print_container(self)

    def __repr__(self):
        content = super(BirchSubcluster, self).__repr__()
        if self.parent:
            parent_repr = 'BirchSubcluster[subcluster_id={}]'\
                    .format(self.parent['cluster_id'])
        else:
            parent_repr = 'None'

        if self.children:
            child_repr = ', '.join([
                  'BirchSubcluster[cluster_id={}]'
                  .format(ctr['cluster_id'])
                  for ctr in self.children])
        else:
            child_repr = ''

        return dedent("""
             {}
               * parent: {}
               * children: [{}]
             """.format(content, parent_repr, child_repr))

    def increment_cluster_id(self, value):
        """ Increment the cluster_id of all children
        by the given value
        """
        self['cluster_id'] += value
        for child in self.children:
            child.increment_cluster_id(value)


def _check_birch_tree_consistency(node):
    """ Check that the _id we added is consistent """
    for el in node.subclusters_:
        if el.samples_id_ is None:
            raise ValueError('Birch was fitted without storing samples. '
                             'Please re-initalize Birch with '
                             'compute_sample_indices=True !')
        if el.n_samples_ != len(el.samples_id_):
            raise ValueError(('For subcluster ',
                             '{}, n_samples={} but len(id_)={}')
                             .format(el, el.n_samples_, el.samples_id_))
        if el.child_ is not None:
            _check_birch_tree_consistency(el.child_)


def _birch_hierarchy_constructor(node, depth=0, cluster_id=0,
                                 container=BirchSubcluster,
                                 prune_single_clusters=True):

    htree = container()
    htree['document_id'] = document_id_list = []
    htree['cluster_id'] = cluster_id
    htree.prune_single_clusters = prune_single_clusters
    # detect if this subcluster has a single child

    for el in node.subclusters_:
        if el.child_ is not None:
            cluster_id += 1
            subtree, cluster_id = _birch_hierarchy_constructor(
                     el.child_, depth=depth+1, cluster_id=cluster_id,
                     container=container,
                     prune_single_clusters=prune_single_clusters)
            if len(subtree.children) == 1 and prune_single_clusters:
                # we are going to skip the single child subcluster,
                # so don't need to increment the cluster_id
                subtree.increment_cluster_id(-1)
                cluster_id += -1
                # skip the single child subcluster
                htree.add_child(subtree.children[0])
            else:
                htree.add_child(subtree)
        else:
            document_id_list += el.samples_id_
    if depth == 0:
        # make sure we return the correct number of clusters
        cluster_id += 1
    return htree, cluster_id


def birch_hierarchy_wrapper(birch, container=BirchSubcluster, validate=True,
                            compute_document_id=True):
    #if not isinstance(birch, Birch):
    #    raise ValueError('the birch object must be created with '
    #                     'freediscovery.cluster.Birch')

    if not hasattr(birch, "root_"):
        raise NotFittedError("The Birch model must be fitted first!")

    if validate:
        _check_birch_tree_consistency(birch.root_)

    htree, n_subclusters = _birch_hierarchy_constructor(birch.root_,
                                                        container=container)
    if validate:
        if len(htree.cluster_id_accumulated) != birch.n_samples_:
            print(htree.cluster_id_accumulated)
            raise ValueError(("Building hierarchy failed: root node contains "
                              "{} documents, while the total document number "
                              "is {}")
                             .format(len(htree.cluster_id_accumulated),
                                     birch.n_samples_))
    if compute_document_id:
        for row in htree.flatten():
            document_id_lst = row.cluster_id_accumulated
            row['cluster_id_accumulated'] = document_id_lst
            row['cluster_size'] = len(document_id_lst)
    return htree, n_subclusters


def _print_container(ctr, depth=0):
    """Print summary clustering hierarchy to stdout."""
    message = "[cluster_id={cluster_id}] N_children: {N_children} N_samples: {N_document}"\
              .format(cluster_id=ctr['cluster_id'],
                      N_children=len(ctr.children),
                      N_document=len(ctr['cluster_id_accumulated']))

    print(''.join(['> ' * depth, message]))

    for child in ctr.children:
        _print_container(child, depth + 1)