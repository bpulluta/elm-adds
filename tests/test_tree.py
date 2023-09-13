# -*- coding: utf-8 -*-
"""
Test
"""
import os
import networkx as nx
from elm import TEST_DATA_DIR
from elm.base import ApiBase
from elm.tree import DecisionTree
import elm.embed

FP_PDF = os.path.join(TEST_DATA_DIR, 'GPT-4.pdf')
FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')

with open(FP_TXT, 'r', encoding='utf8') as f:
    TEXT = f.read()


class MockClass:
    """Dummy class to mock various api calls"""

    chat_messages = [{'role': 'role', 'content': 'test'}]

    @staticmethod
    def chat(prompt):
        """Mock for api.chat()"""
        return prompt


def test_chunk_and_embed(mocker):
    """Simple text to embedding test

    Note that embedding api is mocked here and not actually tested.
    """
    mocker.patch.object(elm.tree.DecisionTree, "api", MockClass)

    graph = nx.DiGraph(text='hello', name='Grant',
                       api=ApiBase(model='gpt-35-turbo'))

    graph.add_node('init', prompt='Say {text} to {name}')
    graph.add_edge('init', 'next', condition=lambda x: 'Grant' in x)
    graph.add_node('next', prompt='How are you?')

    tree = DecisionTree(graph)
    tree.run()

    assert 'init' in tree.history
    assert 'next' in tree.history
