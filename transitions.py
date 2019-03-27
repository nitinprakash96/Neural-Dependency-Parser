#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys


class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        self.sentence = sentence

        self.stack = ["ROOT"]
        self.buffer = [word for word in sentence]
        self.dependencies = []

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """

        if transition == 'S':
            self.stack.append(self.buffer.pop(0))
        elif transition == 'LA':
            dependency, head = self.stack[-2:]
            self.stack.pop(-2)
            self.dependencies.append((head, dependency))
        elif transition == 'RA':
            head, dependency = self.stack[-2:]
            self.stack.pop(-1)
            self.dependencies.append((head, dependency))

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dsependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)

        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []

    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]

    while unfinished_parses:
        minibatch_parses = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch_parses)

        for parse, transition in zip(minibatch_parses, transitions):
            parse.parse_step(transition)
            if len(parse.stack) < 2 and len(parse.buffer) < 1:
                unfinished_parses.remove(parse)

    dependencies = [p.dependencies for p in partial_parses]

    return dependencies
