class Node:
    feature = ""
    children = []

    def __init__(self, feature, children_dict):
        self.addFeature(feature)
        self.generateChildren(children_dict)

    def __str__(self):
        return str(self.feature)

    def addFeature(self, feature):
        self.feature = feature

    def generateChildren(self, children_dict):
        if isinstance(children_dict, dict):
            self.children = children_dict.keys()
