import graphviz

dot = graphviz.Digraph(format='png', engine='dot')
dot.attr(rankdir='TB', nodesep='0.8', ranksep='0.8', splines='spline')
dot.attr('node', fontname='Arial', fontsize='12', penwidth='1.5')
dot.attr('edge', fontname='Arial', fontsize='10', penwidth='1.5')

# Define nodes
dot.node('Data', 'Data', shape='box', style='filled,rounded', fillcolor='#bdf2c3', width='1.2', height='0.8')
dot.node('UserQuery', 'User Query', shape='box', style='filled,rounded', fillcolor='#bdf2c3', width='1.5', height='0.6')
dot.node('SubQ', 'Sub Q tree', shape='box', style='filled,rounded', fillcolor='#bdf2c3', width='1.5', height='0.6')
dot.node('Final', 'Final\nAnswer', shape='box', style='filled,rounded', fillcolor='#bdf2c3', width='1.2', height='0.6')

dot.node('AnsVal', 'Ans\nvalidation\nand progating\nans to root', shape='box', style='filled,rounded', fillcolor='#fbc0c2', width='1.5', height='0.8')
dot.node('Combined', 'combined', shape='diamond', style='filled', fillcolor='#fbc0c2', width='1.5', height='0.8')

dot.node('Semantic', 'semantic leg', shape='diamond', style='filled', fillcolor='#ffeaa7', width='1.5', height='0.8')
dot.node('Lexical', 'lexical\nleg', shape='diamond', style='filled', fillcolor='#ffeaa7', width='1.5', height='0.8')
dot.node('Embedding', 'Embedding', shape='box', style='filled,rounded', fillcolor='#ffeaa7', width='1.5', height='0.6')

dot.node('VectorDB', 'Vector\nDB', shape='cylinder', style='filled', fillcolor='#bdf2c3', width='1.2', height='1.2')
dot.node('GraphDB', 'Graph\nDB', shape='cylinder', style='filled', fillcolor='#bdf2c3', width='1.2', height='1.0')

# Edges
dot.edge('Data', 'Embedding', label='Divided\ninto chunks')
dot.edge('Data', 'GraphDB', label='indexed,\ngraph facts')

dot.edge('Embedding', 'VectorDB', label='Stored\nin Vector\nDB')

dot.edge('UserQuery', 'SubQ', label='LLM')
dot.edge('SubQ', 'Lexical', label='token\nmatching')
dot.edge('SubQ', 'Semantic', label='encoding')

dot.edge('GraphDB', 'Lexical', label='idf\nweighted\ntokens')
dot.edge('VectorDB', 'Semantic', label='Top-k\nsimilar passages/vectors')

dot.edge('Lexical', 'Combined')
dot.edge('Semantic', 'Combined')

dot.edge('Combined', 'AnsVal', label='LLM')
dot.edge('AnsVal', 'SubQ', label='Feedback loop')
dot.edge('AnsVal', 'Final')

dot.render('paper/hamhrag_architecture', view=False)
