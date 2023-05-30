#!/usr/bin/env python3

# nm.py
# Nother Monstrosity - A program language inspired by lisp that is very buggy
# https://github.com/ninjamar/nm
# Version 0.0.14
__version__ = "0.0.15"
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


import argparse
import readline
import sys
import os

# Generic classes/types the interpreter uses


class Ast(list):
    pass


Symbol = str
List = list
String = str
Number = (int, float)
AstType = (Ast, list)
Array = list


class Env(dict):
    """Storage used instead of dict since it allows lookup in "locals" first"""

    def find(self, key: object) -> object:
        """Find a key inside enviornment

        :param key: key to find
        :type key: object
        :return: return value of key lookup
        :rtype: object
        """
        # Use this method instead of "__getitem__" because this method performs a lookup in "locals" then in the main dictionary
        if str(key).split(".")[0] in self["locals"]:
            return self["locals"][key]
        else:
            return self[key]


class Function:
    """Wrapper class to store and execute functions"""

    def __init__(self, engine, name: str, args: list | Ast, code: str) -> None:
        """Class to store and execute functions

        :param name: name of function
        :type name: str
        :param args: list of arguments
        :type args: list | Ast
        :param code: code function runs when executed
        :type code: str
        """
        # Name is only needed for traceback
        self.engine = engine
        self.name = name
        self.args = args
        self.code = code

    def execute(self, givenargs: list | Ast) -> object:
        """Execute a function

        :param givenargs: given arguments of function
        :type givenargs: list | Ast
        :raises TypeError: Wrong number of arguments
        :return: result of executed code
        :rtype: object
        """
        # Wrong arguments given
        if len(givenargs) != len(self.args):
            raise TypeError(
                f"{self.name} expected {len(self.args)} arguments, {len(givenargs)} given"
            )
        # Apply arguments to "locals"
        self.engine.env["locals"] = dict(zip(self.args, givenargs))
        # Evaluate function
        result = self.engine.evaluater(self.code)
        # Clear "locals"
        self.engine.env["locals"] = {}
        return result

    def __call__(self, *givenargs) -> object:
        """Execute a function

        :return: result of executed code
        :rtype: object
        """
        return self.execute(givenargs)


class Engine:
    """Class to run NM Code"""

    def __init__(self, importpath=None, env=None):
        """Class to run NM Code

        :param importpath: path of module imports, defaults to None
        :type importpath: _type_, optional
        :param env: env to use, defaults to None
        :type env: _type_, optional
        """
        self.path = []
        self.modules = {}
        # Create a main global enviornment
        if env is None:
            self.env = self.default_env()
        else:
            self.env = env
        if importpath is None:
            self.path.append(os.getcwd())
        else:
            self.path.append(importpath)
        sys.path.append(self.path[0])

    # Tokenize a program but don't apply corrections
    # TLDR; Transform a string into a list
    def tokenize(self, chars: str) -> str:
        """Turn a string into tokens

        :param chars: string to tokenize
        :type chars: str
        :return: tokenized string
        :rtype: str
        """
        tokens = chars.replace("(", " ( ").replace(")", " ) ").split()
        tokens.insert(0, "(")
        tokens.append(")")
        return tokens

    # Apply some corrections to generated tokens
    # Specifically this allows strings with whitespace inside of them to be parsed as a single item rather than many
    def fix(self, tokens: list) -> list:
        """Apply miscellaneous fixes

        :param tokens: tokens generated by tokenize
        :type tokens: list
        :return: list of fixed tokens
        :rtype: list
        """
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
    def generate_ast(self, tokens: list) -> list:
        """Generate an ast tree from a list of tokens

        :param tokens: list of tokens for ast to generate from
        :type tokens: list
        :raises SyntaxError: Unexpected EOF
        :raises SyntaxError: Unexpected )
        :return: ast tree of tokens
        :rtype: list
        """
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
                    g = self.generate_ast(tokens)
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
                return self.atom(token)

    # Convert a token into a type
    # Current hierchy is from int -> float -> Symbol
    # String data type isn't present because at this stage strings have no effect
    def atom(self, token: str) -> object:
        """Convert a token into their proper type

        :param token: token
        :type token: str
        :return: converted token
        :rtype: object
        """
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return Symbol(token)

    # Perform tokenization, fixes and ast generation in one step
    def parse(self, program: str) -> str:
        """Perform tokenization, fixes and ast generation

        :param program: string of raw tokens that are unparsed
        :type program: str
        :return: ast tree
        :rtype: str
        """
        # return self.generate_ast(self.fix(self.tokenize(self.handlecomments(program))))
        # return self.generate_ast(self.fix(self.tokenize(self.handlecomments([program] if len(program) == 1 else self.handlecomments(program)))))
        return self.generate_ast(self.fix(self.tokenize(program)))

    # Large function to create an enviorment/standard library
    def default_env(self) -> Env:
        """Create a default env

        :return: env
        :rtype: Env
        """
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
                "chr": lambda a: chr(a),
                "ord": lambda a: ord(a),
                "print": lambda *a: print(*a),
                "quit": lambda a: sys.exit(a),
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

    # Migrated to function
    # def create_function(name, args, code):
    #    return lambda *a: _lazy_execute_function(name, code, a, args)

    # Import library into env
    def importmodule(self, library: str) -> None:
        """Import a module into env

        :param library: name of library to import
        :type library: str
        :raises ModuleNotFoundError: Module cannot be found
        """
        try:
            # Firstly try finding module as NM
            # Currently module can't handle circular imports
            # Evaluated code is added to "env"
            # self.execfile(f"{library}.nm")
            path = None
            for _path in self.path:
                if os.path.isdir(_path):
                    path = _path

            # Path doesn't exist or self.path is empty
            if path is None:
                raise FileNotFoundError
            module = Engine()
            module.execfile(path + "/" + library + ".nm")
            self.modules[library] = module.env
            self.env.update({library + "." + k: v for k, v in module.env.items()})
        except FileNotFoundError:
            try:
                # Import python module
                module = vars(__import__(library))
                self.modules[library] = module
                self.env.update({library + "." + k: v for k, v in module.items()})
            except:
                raise ModuleNotFoundError(f"Module {library} unable to be resolved")

    def get_type(self, ast: Ast) -> type:
        """Get type from ast

        :param ast: ast to get type from
        :type ast: Ast
        :return: type of ast
        :rtype: type
        """
        if isinstance(ast, AstType):
            return type(ast[0])
        else:
            return type(ast)

    def isstr(self, ast: Ast) -> String | None:
        """Check if ast is a string

        :param ast: ast to check
        :type ast: Ast
        :return: ast as string else None
        :rtype: String | None
        """
        if isinstance(ast, AstType) and len(ast) == 1 and type(ast[0]) == str:
            if ast[0][0] == '"' and ast[0][-1] == '"':
                return String(ast[0][1:-1])
        elif type(ast) == str and ast[0] == '"' and ast[-1] == '"':
            return String(ast[1:-1])
        return None

    def isarr(self, ast: Ast) -> Array | None:
        """Check if ast is an array

        :param ast: ast to check
        :type ast: Ast
        :return: ast as array else none
        :rtype: Array | None
        """
        if isinstance(ast, AstType) and len(ast) == 1 and type(ast[0]) == str:
            if ast[0][0] == "[" and ast[0][-1] == "]":
                return Array(ast[0][1:-1].split(","))
        elif type(ast) == str and ast[0] == "[" and ast[-1] == "]":
            return Array(ast[1:-1].split(","))
        return None

    # Evaluate a single node
    def evaluate(self, ast: Ast) -> object | None:
        """Evaluate a single node

        :param ast: ast to be evaluated
        :type ast: Ast
        :param env: enviornment to be used, defaults to env
        :type env: Env, optional
        :return: result of evaluated node
        :rtype: object | None
        """

        # Ast shouldn't be empty
        if ast == []:
            return ast
        # See comment below
        if isinstance(ast, Number):
            return ast
        # Forgot what this does but it is VERY important
        if len(ast) == 1 and isinstance(ast, AstType):
            if isinstance(ast[0], Number):
                return ast[0]
        # Check if ast is str
        if (isstr := self.isstr(ast)) is not None:
            return isstr
        # Check if ast is array
        if (isarr := self.isarr(ast)) is not None:
            return isarr
        # Somehow important
        if not isinstance(ast, AstType):
            return self.env.find(ast)

        match ast:
            # Self explanatory
            # Import module
            case ["include", library]:
                self.importmodule(library)
                return
            # Return value from a function
            # Might be an issue evaluating value
            case ["return", *value]:
                self.env["flags"]["FRETURN"] = True
                return self.evaluate(value)
            # For each loop over item and call function every iteration
            case ["each", item, function]:
                item = self.env.find(item)
                function = self.env.find(function)
                for i in item:
                    function(i)
                return
            # Hasn't been fully tested but probably doesn't work
            # Conditionals might not work because it hasn't been tested
            case ["if", test, consequence]:
                if self.evaluate(test):
                    return self.evaluate(
                        consequence
                    )  # Can only evaluate singluar expressions, to change to muliple use self.evaluater
                return
            case ["if", test, consequence, alternative]:
                if self.evaluate(test):
                    return self.evaluate(consequence)
                else:
                    return self.evaluate(alternative)
            # Define/set a variable
            case ["define", symbol, exp]:
                self.env[symbol] = self.evaluate(exp)
                return
            # Special list operations to modify list
            # These haven't been tested
            case ["append", symbol, item]:
                self.env.find(symbol).append(self.evaluate(item))
                return
            case ["remove", symbol, item]:
                self.env.find(symbol).remove(self.evaluate(item))
                return
            case ["extend", symbol, item]:
                self.env.find(symbol).extend(self.env.find(self.evaluate(item)))
                return
            case ["clear", symbol]:
                self.env.find(symbol).clear()
                return
            case ["index", symbol, item]:
                return self.env.find(symbol).index(self.evaluate(item))
            case ["insert", symbol, index, item]:
                self.env.find(symbol).insert(index, self.evaluate(item))
                return
            case ["pop", symbol, index]:
                return self.env.find(symbol).pop(index)
            case ["reverse", symbol]:
                self.env.find(symbol).reversed()
                return
            case ["sort", symbol]:
                self.env.find(symbol).sort()
                return
            case ["count", symbol, item]:
                return self.env.find(symbol).count(item)
            # Define a function
            case ["func", name, args, code]:
                self.env[name] = Function(self, name, args, code)
                return
            # Everything else which is also where things get complicated/confusing
            case _:
                # SUPER important
                if len(ast) == 1:
                    return self.env.find(ast[0] if isinstance(ast, AstType) else ast)

                try:
                    # Get function to run
                    proc = self.env.find(ast[0])  # self.evaluate(ast[0])
                    # Handle arguments (probably the most confusing thing)
                    args = (self.evaluate(arg) for arg in ast[1])
                # Not sure why catch is here but it helped fix a problem with the arguments of a function
                except (TypeError, KeyError):
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
    def evaluater(self, ast: Ast, main=False) -> object:
        """Evaluate a list of nodes

        :param ast: ast to evaluate
        :type ast: Ast
        :param main: is program running as main?, defaults to False
        :type main: bool, optional
        :return: result of executed function, defaults to None
        :rtype: object | None
        """
        # Handle comments
        if ast == []:
            return
        for node in ast:
            result = self.evaluate(node)
            # Make sure we aren't main (main can't return)
            # Determine if we need to return
            if not main and self.env["flags"]["FRETURN"]:
                return result
            # In the future, copy all the flags so they don't all have to be put here
            self.env["flags"] = {"FRETURN": False}
        return result

    def handlecomments(self, program: list) -> str:
        """Handle comments

        :param program: program
        :type program: list
        :return: program but with comments removed
        :rtype: str
        """
        # result = []
        result = ""
        for token in program:
            if token[0] == ";":
                continue
            elif ";" in token:
                # result.append(token[:token.index(";")])
                result += token[: token.index(";")]
            else:
                # result.append(token)
                result += token
        return result

    # Execute a NM program from a file
    def execfile(
        self, fname: str, show_ast: bool = False, no_eval: bool = False
    ) -> None:
        """Execute an NM program from a file

        :param fname: path to file
        :type fname: str
        :param show_ast: print ast, defaults to False
        :type show_ast: bool, optional
        :param no_eval: disable evaluation, defaults to False
        :type no_eval: bool, optional
        """
        with open(fname) as f:
            contents = (
                f.readlines()
            )  # This is required because self.handlecomments checks line by line
        parsed = self.parse(self.handlecomments(contents))
        # parsed = self.parse(contents)
        if show_ast:
            print(parsed)
        # Main loop
        if not no_eval:
            self.evaluater(parsed, main=True)

    def execstr(self, code: str) -> object:
        """Wrapper to run code

        :param code: code to run
        :type code: str
        :return: result of evaluated expression
        :rtype: object
        """
        parsed = self.parse(self.handlecomments([code]))
        # parsed = self.parse(code)
        return self.evaluater(parsed, main=True)

    def execstrfromcli(
        self, code: str, show_ast: bool = False, no_eval: bool = False
    ) -> None:
        """Wrapper to run code from a cli program

        :param code: code to run
        :type code: str
        :param show_ast: print ast, defaults to False
        :type show_ast: bool, optional
        :param no_eval: disable evaluation, defaults to False
        :type no_eval: bool, optional
        """
        parsed = self.parse(self.handlecomments([code]))
        # parsed = self.parse(code)
        if show_ast:
            print(parsed)
        if not no_eval:
            self.evaluater(parsed, main=True)

    def repl(self) -> None:
        """Run an interactive session"""
        print((f"NM {__version__}"))
        while True:
            try:
                cmd = input("nm>")
                ret = self.execstr(cmd)
                if ret is not None:
                    print(ret)
            except Exception as e:
                self.show_exeption("repl", e)

    def show_exeption(self, module: str, exc: Exception) -> None:
        """Show an exception raised from the program

        :param module: location where exception occured
        :type module: str
        :param exc: instance of exception
        :type exc: Exception
        """
        match exc:
            case KeyError():
                print(f"<{module}>: Illegal symbol {exc.args[0]}")
            case SyntaxError():
                print(f"<{module}>: Unexpected EOF")
            case EOFError():
                self.evaluate(Ast(["quit", [130]]))
            case _:
                raise exc


def cli() -> None:
    """Handle cli arguments

    :raises argparse.ArgumentError: invalid arguments
    """
    parser = argparse.ArgumentParser(
        description="Execute a nm file", epilog="Thank you for using %(prog)s! :)"
    )
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--ast", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--expr")
    args = parser.parse_args()

    if args.expr and args.path:
        raise argparse.ArgumentError("Cannot use --expr and path together")
    elif args.path is None and args.expr is None:
        engine = Engine()
        try:
            engine.repl()
        except KeyboardInterrupt:
            engine.evaluate(Ast(["quit", [130]]))
    elif args.expr:
        engine = Engine()
        engine.execstrfromcli(args.expr, args.ast, args.no_eval)
    else:
        engine = Engine(os.path.dirname(os.path.abspath(args.path)))
        engine.execfile(args.path, args.ast, args.no_eval)


if __name__ == "__main__":
    cli()
