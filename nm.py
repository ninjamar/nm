#!/usr/bin/env python3

# Copyright 2023 ninjamar
# nm.py
# Nother Monstrosity - A program language inspired by lisp that is very buggy
# Version 0.0.3

# MIT License
#
# Copyright (c) 2023 ninjamar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# I adapted some functions (tokenize, generate_ast, atom) from Peter Norvig's lispy 
## (c) Peter Norvig, 2010-16; See http://norvig.com/lispy.html


# Generic classes/types the interpreter uses

class Ast(list):
    pass

Symbol = str
List = list
String = str
Number = (int, float)


# Storage used instead of dict since it allows lookup in "locals" first
class Env(dict):
    def find(self, key):
        # Use this method instead of "__getitem__" because this method performs a lookup in "locals" then in the main dictionary
        if key in self["locals"]:
            return self["locals"][key]
        else:
            return self[key]


# Tokenize a program but don't apply corrections
# TLDR; Transform a string into a list
def tokenize(chars: str) -> str:
    return chars.replace("(", " ( ").replace(")", " ) ").split()


# Apply some corrections to generated tokens
# Specifically this allows strings with whitespace inside of them to be parsed as a single item rather than many
def fix(tokens: list) -> list:
    # correct a list to allow for types such as strings
    new_tokens = []
    # Using next is a more efficient solution instead of iterating over tokens because we can perform operations on multiple tokens at once
    tokens = iter(tokens)
    while True:
        try:
            token = next(tokens)
            # Start of string literal
            if token[0] == '"':
                # String literal might be self contained
                if token[-1] == '"':
                    new_tokens.append(token)
                    continue
                else:
                    # String literal is split across multiple tokens
                    next_token = next(tokens)
                    # Fetch all tokens until we find end of string
                    # Currently this is untested on prorgrams with unclosed string literals
                    while next_token[-1] != '"':
                        # Append space here because tokenize removes all spaces
                        token += " " + next_token
                        next_token = next(tokens)
                    token += " " + next_token
                    new_tokens.append(token)
            else:
                # Token isn't a string
                new_tokens.append(token)
        except StopIteration:
            break
    return new_tokens


# Generate the ast from a list of tokens
def generate_ast(tokens: list) -> list:
    if len(tokens) == 0:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)  # remove starting from the right
    match token:
        case "(":
            # Recursively generate ast
            r = []
            while (
                tokens[0] != ")"
            ):  # we pop off each item so list shifts left each iteration
                g = generate_ast(tokens)
                if isinstance(g, list):
                    # Possibly for legacy reasons
                    g = Ast(g)
                r.append(g)
            # Pop off ")"
            tokens.pop(0)
            # Return the tokens but as an ast object for easy type checking
            return Ast(r)
        case ")":
            raise SyntaxError("Unexpected )")
        case _:
            # 99% of the time this probably gets called
            return atom(token)


# Convert a token into a type
# Current hierchy is from int -> float -> Symbol
# String data type isn't present because at this stage strings have no effect
def atom(token: str) -> object:
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


# Perform tokenization, fixes and ast generation in one step
def parse(program: str) -> str:
    return generate_ast(fix(tokenize(program)))


# Large function to create an enviorment/standard library
def default_env():
    # Standard library functions
    env = Env(
        {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "mod": lambda a, b: a % b,
            "exp": lambda a, b: a**b,
            "floordiv": lambda a, b: a // b,
            "gt": lambda a, b: a > b,
            "gte": lambda a, b: a >= b,
            "lt": lambda a, b: a < b,
            "lte": lambda a, b: a <= b,
            "eq": lambda a, b: a == b,
            "print": lambda *a: print(*a),
            # All variables defined inside functions
            "locals": {},
            # Flags used by the interpreter
            "flags": {
                # Determine if function returns and therefore should loop return
                "FRETURN": False
            },
        }
    )
    # Handle aliases
    env.update(
        {
            "+": env["add"],
            "-": env["sub"],
            "*": env["mul"],
            "/": env["div"],
            "%": env["mod"],
            "**": env["exp"],
            "//": env["floordiv"],
            ">": env["gt"],
            ">=": env["gte"],
            "<": env["lt"],
            "<=": env["lte"],
            "==": env["eq"],
        }
    )
    return env


# Create a main global enviornment
global_env = default_env()

# Migrated to function
# def create_function(name, args, code):
#    return lambda *a: _lazy_execute_function(name, code, a, args)


# Wrapper class to store and execute functions
class Function:
    def __init__(self, name, args, code):
        # Name is only needed for traceback
        self.name = name
        self.args = args
        self.code = code

    def execute(self, givenargs):
        # Wrong arguments given
        if len(givenargs) != len(self.args):
            raise TypeError(
                f"{self.name} expected {len(self.args)} arguments, {len(givenargs)} given"
            )
        # Apply arguments to "locals"
        global_env["locals"] = dict(zip(self.args, givenargs))
        # Evaluate function
        result = evaluater(self.code)
        # Clear "locals"
        global_env["locals"] = {}
        return result

    def __call__(self, *givenargs):
        return self.execute(givenargs)


# Import library into env
def importmodule(library):
    try:
        # Firstly try finding module as NM
        # Currently module can't handle circular imports
        # Evaluated code is added to "global_env"
        execfile(f"{library}.nm")
    except FileNotFoundError:
        try:
            # Import python module
            global_env.update(vars(__import__(library)))
        except:
            raise ModuleNotFoundError(f"Module {library} unable to be resolved")


# Evaluate a single node
def evaluate(ast, env=global_env):
    # Check type of node
    if isinstance(ast, Number):
        return ast
    elif isinstance(ast, Symbol):
        return env.find(ast)
    
    # Really hacky solution to figure out if function is string...
    # Currently this function is recursive so we get MANY different formats of inputs which tend to break stuff
    if len(ast) == 1:
        if ast[0][0] == '"' and ast[0][-1] == '"':
            return String(ast[0][1:-1])
        else:
            # I somehow broke NM
            return env.find(ast[0])

    # Match-case pattern is actually useful
    match ast:
        # Self explanatory
        # Import module
        case ["include", library]:
            importmodule(library)
            return
        # Return value from a function
        # Might be an issue evaluating value
        case ["return", *value]:
            global_env["flags"]["FRETURN"] = True
            return evaluate(value)
        # Hasn't been fully tested but probably doesn't work
        # Conditionals might not work because it hasn't been tested
        case ["if", test, consequence, alternative]:
            if evaluate(test, env):
                return evaluate(consequence, env)
            else:
                return evaluate(alternative, env)
        # Define/set a variable
        case ["define", symbol, exp]:
            env[symbol] = evaluate(exp, env)
            return
        # Define a function
        case ["func", name, args, code]:
            env[name] = Function(name, args, code)
            return
        # Everything else which is also where things get complicated/confusing
        case _:
            # Not sure what this does but it is important
            if len(ast) > 2 or len(ast) <= 1:
                return ast[0] if len(ast) == 1 else ast

            try:
                # Get function to run
                proc = evaluate(ast[0])
                # Handle arguments (probably the most confusing thing)
                args = (evaluate(arg, env) for arg in ast[1])
            # Not sure why catch is here but it helped fix a problem with the arguments of a function
            except TypeError:
                return ast
            # Arguments have been evaluated but functions haven't been called
            evaluated_args = []
            while True:
                try:
                    arg = next(args)
                    # Check if arg is a function
                    if callable(arg):
                        new_arg = next(args)
                        # Make sure argument is a list
                        if not isinstance(new_arg, list):
                            # Somehow this handles strings...
                            new_arg = [new_arg]
                        evaluated_args.append(arg(*new_arg))
                    else:
                        evaluated_args.append(arg)
                except StopIteration:
                    break
            # Actually call function
            result = proc(*evaluated_args)
            return result


# Main loop for evaluation
# Main argument should be True if calling directly since functions use this to evaluate code
def evaluater(ast, main=False):
    for node in ast:
        result = evaluate(node)
        # Make sure we aren't main (main can't return)
        # Determine if we need to return
        if not main and global_env["flags"]["FRETURN"]:
            return result
        # In the future, copy all the flags so they don't all have to be put here
        global_env["flags"] = {"FRETURN": False}


# Execute a NM program from a file
def execfile(fname):
    with open(fname) as f:
        contents = f.read()
    parsed = parse(contents)
    # Main loop
    evaluater(parsed, main=True)


if __name__ == "__main__":
    # TODO - Implement an actual cli
    # TODO - Add type hinting for all functions
    # TODO - Add comments
    execfile("test.nm")
