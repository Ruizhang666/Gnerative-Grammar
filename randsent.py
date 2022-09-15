#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

import re
import numpy as np
from binarytree import Node

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file

        Returns:
            self
        """
        # Parse the input grammar file
        results = self._load_rules_from_file(grammar_file)
        self.sentence_type = results[0]
        self.sentence_prob = results[1]
        self.rules = results[2]
        self.rules_prob = results[3]
        self.vocab = results[4]
        self.vocab_prob = results[5]

        # These are used for section2 and section3
        self.tree_structure = None
        self.pointer = None
        self.max_expansions = None
        self.output_s = ""
        self.output_type = None

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file
        """
        with open(grammar_file, encoding="utf-8") as f:
            read_data = f.read()
        f.closed
        sentence_type = []
        sentence_prob = []
        nonterminal = ["S", "NP", "VP", "Verb", "Det", "Noun", "PP", "Prep", "Adj"]
        rules = {}
        rules_prob = {}
        vocab = {}
        vocab_probability = {}
        make_list = re.split("\n\n |#", read_data)
        for i in make_list:
            x = re.search("\t", i)
            if x:
                info = i.split("\n")
                for j in info:
                    if "ROOT" in j:
                        info_list = j.split("\t")
                        sentence_type.append(info_list[2])
                        sentence_prob.append(info_list[0])
                    elif "\t" in j:
                        derivation = j.split("\t")[2]
                        check_its_nature = derivation.split(" ")[0]
                        if re.search("\t", j) and check_its_nature in nonterminal:
                            if j.split("\t")[1] not in rules:
                                rules[j.split("\t")[1]] = [derivation.split(" ")]
                                rules_prob[j.split("\t")[1]] = [j.split("\t")[0]]
                            else:
                                rules[j.split("\t")[1]].append(derivation.split(" "))
                                rules_prob[j.split("\t")[1]].append(j.split("\t")[0])
                        else:
                            if j.split("\t")[1] not in vocab:
                                vocab[j.split("\t")[1]] = derivation.split(" ")
                                vocab_probability[j.split("\t")[1]] = [j.split("\t")[0]]
                            else:
                                vocab[j.split("\t")[1]].append(j.split("\t")[2])
                                vocab_probability[j.split("\t")[1]].append(j.split("\t")[0])
        return [sentence_type, sentence_prob, rules, rules_prob, vocab, vocab_probability]

    def helper1(self, max_expansions, start_symbol):

        def compute_probability_distribution(odds_list):
            total = 0
            out = []
            for i in odds_list:
                total += float(i)
            for i in odds_list:
                out.append(float(i) / total)
            return out

        nonterminals = list(self.rules.keys())
        terminals = list(self.vocab.keys())

        if start_symbol == "ROOT":
            self.tree_structure = Node("S")
            self.pointer = self.tree_structure
            self.max_expansions = max_expansions - 1
            prob1 = compute_probability_distribution(self.sentence_prob)
            s_type = int(np.random.choice(list(range(len(self.sentence_type))), 1, prob1))
            resulting_sentence = self.sentence_type[s_type]
            self.output_type = resulting_sentence
            self.helper1(self.max_expansions, "S")
        elif (start_symbol in terminals) and (start_symbol in nonterminals):
            odds_list = self.vocab_prob[start_symbol] + self.rules_prob[start_symbol]
            dis = compute_probability_distribution(odds_list)
            sampled_expansion = int(np.random.choice(len(odds_list), 1, dis))
            if sampled_expansion <= (len(self.vocab_prob) - 1):
                sampled_vocab = self.vocab[start_symbol][sampled_expansion]
                self.output_s = self.output_s + " " + sampled_vocab
                self.pointer.left = Node(sampled_vocab)
            else:
                sampled_rule = self.rules[start_symbol][sampled_expansion - len(self.vocab_prob[start_symbol])]
                left_child = sampled_rule[0]
                right_child = sampled_rule[1]
                # start making expansion as we normally would
                self.max_expansions = self.max_expansions - 1
                temp = self.pointer
                if self.max_expansions >= 0:
                    self.pointer.left = Node(left_child)
                    self.pointer = self.pointer.left
                    self.helper1(self.max_expansions, left_child)
                else:
                    self.output_s = self.output_s + " " + "..."
                    self.pointer.left = Node("...")
                self.max_expansions = self.max_expansions - 1
                if self.max_expansions >= 0:
                    self.pointer = temp
                    self.pointer.right = Node(right_child)
                    self.pointer = self.pointer.right
                    self.helper1(self.max_expansions, right_child)
                else:
                    self.pointer = temp
                    self.output_s = self.output_s + " " + "..."
                    self.pointer.right = Node("...")

        elif start_symbol in nonterminals:
            extract_rules = self.rules[start_symbol]
            if len(extract_rules) > 1:
                prob2 = compute_probability_distribution(self.rules_prob[start_symbol])
                sample_rules = int(np.random.choice(len(extract_rules), 1, prob2))
                rule_used = extract_rules[sample_rules]
            else:
                rule_used = extract_rules[0]
            left_child = rule_used[0]
            right_child = rule_used[1]
            # Note that we can only deduct the max_expansion on if the new child is nonterminal
            self.max_expansions = self.max_expansions - 1
            temp = self.pointer
            if self.max_expansions >= 0:
                self.pointer.left = Node(left_child)
                self.pointer = self.pointer.left
                self.helper1(self.max_expansions, left_child)
            else:
                self.output_s = self.output_s + " " + "..."
                self.pointer.left = Node("...")
            self.max_expansions = self.max_expansions - 1
            if self.max_expansions >= 0:
                self.pointer = temp
                self.pointer.right = Node(right_child)
                self.pointer = self.pointer.right
                self.helper1(self.max_expansions, right_child)
            else:
                self.pointer = temp
                self.output_s = self.output_s + " " + "..."
                self.pointer.right = Node("...")
        else:
            prob3 = compute_probability_distribution(self.vocab_prob[start_symbol])
            words = self.vocab[start_symbol]
            sampled_word = str(np.random.choice(words, 1, prob3)[0])
            self.output_s = self.output_s + " " + sampled_word
            self.pointer.left = Node(sampled_word)

    def treeToString(self, root: Node, string: list):

        if self.tree_structure is None:
            return

        string.append(str(root.value))

        if not root.left and not root.right:
            return

        string.append('(')
        self.treeToString(root.left, string)
        string.append(')')

        if root.right:
            string.append('(')
            self.treeToString(root.right, string)
            string.append(')')

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent
                the tree (using bracket notation) that records how the sentence
                was derived

            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """

        # helper
        def check(string):
            nonterminals = list(self.rules.keys())
            terminals = list(self.vocab.keys())
            if string == "(" or string == ")":
                return False
            elif string == "...":
                return True
            elif (string not in nonterminals) and (string not in terminals):
                return True
            else:
                return False

        self.helper1(max_expansions, start_symbol)
        output_sentence = self.output_type.replace("S",self.output_s)

        if derivation_tree == True:
            words_list = []
            self.treeToString(self.tree_structure, words_list)
            b_copy = list(words_list)
            for i in range(len(words_list)):
                if check(words_list[i]):
                    b_copy[i - 1] = ""
                    b_copy[i + 1] = ""
            out = ""
            for i in b_copy:
                out = out + " " + i
            sen1 = "(ROOT " + self.output_type + " )"
            out="( "+out+" )"
            print(sen1.replace("S", out))

        return output_sentence


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
