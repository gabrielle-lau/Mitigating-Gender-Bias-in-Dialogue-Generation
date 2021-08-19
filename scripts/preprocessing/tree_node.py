"""Tree class for storing Reddit thread"""
class Node(dict):
    def __init__(self, uid):
        self._parent = None  # pointer to parent Node
        self['id'] = uid  # keep reference to id #            
        self['children'] = [] # collection of pointers to child Nodes
        self.data = None

    @property
    def parent(self):
        # simply return the object at the _parent pointer
        return self._parent  

    @parent.setter
    def parent(self, node):
        self._parent = node
        # add this node to parent's list of children
        node['children'].append(self)    